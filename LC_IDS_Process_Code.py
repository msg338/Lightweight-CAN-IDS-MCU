#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCU 기반 LC-IDS 전처리(사용자 lcids_preproc.c/h와 1:1 동일 규칙) + 2D CNN 학습 + TFLite(FP32) 변환
+ TFLite RAW 정확도 평가(Confusion Matrix)
+ Attack/Normal flag 분포 확인 + flag 기반 라벨 정교화(옵션)
+ MCU main.c 호환 segmented can_frames_dump.h 생성 (g_can_dump_segments 방식)

핵심 조건
- ID: 29bit 마스킹 (0x1FFFFFFF)
- ID bits: MSB-first (bit28..bit0)
- DATA: DLC 이후 바이트 0으로 마스킹
- DATA bits: byte0..7, 각 바이트 MSB-first
- Window: N_SEQ=7
- 입력: F1(7x29) + F2(8x8) → padding → 2채널 (8x29x2) 단일 입력
  - F1: (7,29) → zero-pad 1행 → (8,29,ch0)
  - F2: (8,8)  → zero-pad 21열 → (8,29,ch1)
- 데이터: Kaggle "pranavjha24/car-hacking-dataset" (정상: txt, 공격: csv)

수정 사항
- [FIX 1] 공격 라벨 정제 무력화 버그 수정
- [FIX 2] EarlyStopping restore_best_weights=True 로 변경
- [FIX 3] MONITOR_METRIC: val_auc → val_acc
- [FIX 4] 출력 레이어 sigmoid → softmax (2-class)
"""

import os
import re
import glob
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# 0) 설정
# =========================
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

N_SEQ = 7

BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

MAX_FRAMES_NORMAL = 800_000   # Normal 프레임 수 제한 (988K까지 늘릴 수 있음)
MAX_FRAMES_ATTACK_TOTAL = 800_000

ATTACK_TYPES = ["DoS", "Fuzzy", "RPM", "Gear"]

OUT_H5 = "lcids_mcu_fp32_2d_last.h5"
BEST_H5 = "lcids_mcu_fp32_2d_best.h5"
OUT_TFLITE_FP32 = "lcids_mcu_fp32_2d_best.tflite"

EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE = 2
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6

USE_FLAG_LABELING = True

EXPORT_DUMMY_SEGMENTED_H = True
DUMMY_SEG_H_OUT = "can_frames_dump.h"
DUMMY_PER_SEG_MAX_FRAMES = 2500

MONITOR_METRIC = "val_acc"
MONITOR_MODE = "max"

# =========================
# 1) Kaggle 데이터 다운로드
# =========================
def download_carhacking():
    import kagglehub
    return kagglehub.dataset_download("pranavjha24/car-hacking-dataset")

# =========================
# 2) Normal TXT 파서
# =========================
NORMAL_RE = re.compile(
    r"Timestamp:\s*(?P<ts>[0-9]+\.[0-9]+)\s+ID:\s*(?P<id>[0-9A-Fa-f]+)\s+.*?DLC:\s*(?P<dlc>[0-9]+)\s+(?P<data>(?:[0-9A-Fa-f]{2}\s+){0,7}[0-9A-Fa-f]{2})\s*$"
)

def parse_normal_line(line: str):
    line = line.strip()
    if not line:
        return None

    m = NORMAL_RE.match(line)
    if not m:
        return "NO_MATCH"

    ts = float(m.group("ts"))
    cid = m.group("id")
    dlc = int(m.group("dlc"))

    data_tokens = re.split(r"\s+", m.group("data").strip())
    b = [int(x, 16) for x in data_tokens[:8]]
    b = b[:8] + [0] * (8 - len(b))

    return [ts, cid, dlc] + b + ["N"]

def read_normal_txt(path: str, max_bad_ratio=0.001):
    rows = []
    bad = 0
    total = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            r = parse_normal_line(line)

            if r is None:
                continue

            if r == "NO_MATCH":
                bad += 1
                if bad <= 5:
                    print("[NORMAL NO_MATCH sample]", line.rstrip("\n"))
                continue

            rows.append(r)

    if len(rows) == 0:
        raise RuntimeError("normal txt 파싱 결과 0행입니다(확인 필요).")

    bad_ratio = bad / max(1, total)
    print(f"[Normal TXT parse] total={total}, rows={len(rows)}, bad={bad}, bad_ratio={bad_ratio:.6f}")

    if bad_ratio > max_bad_ratio:
        raise RuntimeError("normal txt 파싱 실패 비율이 큽니다(확인 필요).")

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "id", "dlc", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "flag"]
    )
    return df

# =========================
# 3) Attack CSV 로드
# =========================
ATTACK_COLS = ["timestamp", "id", "dlc"] + [f"d{i}" for i in range(8)] + ["flag"]
HEX_CHARS = set("0123456789abcdefABCDEF")

def parse_byte_token(tok) -> int:
    if tok is None or (isinstance(tok, float) and np.isnan(tok)):
        return 0

    s = str(tok).strip()
    if s == "":
        return 0

    if len(s) <= 2 and all(c in HEX_CHARS for c in s):
        return int(s, 16)

    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return 0

def read_attack_csv(path: str):
    return pd.read_csv(path, header=None, names=ATTACK_COLS)

# =========================
# 4) 표준화 + DLC 마스킹
# =========================
def standardize_and_mask(df: pd.DataFrame):
    df = df.copy()

    df["id"] = df["id"].astype(str).str.strip()
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(8).astype(np.int32)
    df["dlc"] = df["dlc"].clip(0, 8).astype(np.int32)

    for i in range(8):
        c = f"d{i}"
        df[c] = (
            df[c]
            .map(parse_byte_token)
            .astype(np.int32)
            .clip(0, 255)
            .astype(np.uint8)
        )

    dlc = df["dlc"].to_numpy(dtype=np.int32)
    data = df[[f"d{i}" for i in range(8)]].to_numpy(dtype=np.uint8)
    mask = (np.arange(8, dtype=np.int32)[None, :] >= dlc[:, None])
    data[mask] = 0
    df[[f"d{i}" for i in range(8)]] = data

    df["flag"] = df["flag"].fillna("").astype(str).str.strip()
    df["flag"] = df["flag"].replace("nan", "")

    return df

# =========================
# 5) MCU 전처리(동일 규칙)
# =========================
def mask29_u32(x: int) -> int:
    return int(x) & 0x1FFFFFFF

def id_to_int29(id_str: str) -> int:
    s = str(id_str).strip()

    if s == "" or s.lower() == "nan":
        return 0

    try:
        if s.lower().startswith("0x"):
            v = int(s, 16)
        else:
            v = int(s, 10)
    except Exception:
        try:
            v = int(s, 16)
        except Exception:
            v = 0

    return mask29_u32(v)

def id29_to_bits29(id29: int) -> np.ndarray:
    id29 = mask29_u32(id29)
    bits = np.empty((29,), dtype=np.float32)

    for i in range(29):
        bit = (id29 >> (28 - i)) & 1
        bits[i] = 1.0 if bit else 0.0

    return bits

def data8_to_bits64(data8: np.ndarray) -> np.ndarray:
    out = np.empty((64,), dtype=np.float32)
    idx = 0

    for b in range(8):
        v = int(data8[b])
        for k in range(8):
            bit = (v >> (7 - k)) & 1
            out[idx] = 1.0 if bit else 0.0
            idx += 1

    return out

def mcu_preproc_one_window_2ch(id_seq_int29: np.ndarray, last_data8: np.ndarray):
    """
    F1(7x29) + F2(8x8) → 2채널 (8x29x2) 단일 텐서
    ch0: F1 zero-pad 1행 아래 → (8,29)
    ch1: F2 zero-pad 21열 오른쪽 → (8,29)
    """
    # F1: (7,29) → (8,29) zero-pad 마지막 행
    f1_flat = np.empty((N_SEQ * 29,), dtype=np.float32)
    for t in range(N_SEQ):
        f1_flat[t * 29:(t + 1) * 29] = id29_to_bits29(int(id_seq_int29[t]))
    ch0 = np.zeros((8, 29), dtype=np.float32)
    ch0[:N_SEQ, :] = f1_flat.reshape((N_SEQ, 29))  # 마지막 행은 0

    # F2: (8,8) → (8,29) zero-pad 오른쪽 21열
    f2_flat = data8_to_bits64(last_data8.astype(np.uint8))
    ch1 = np.zeros((8, 29), dtype=np.float32)
    ch1[:, :8] = f2_flat.reshape((8, 8))  # 오른쪽 21열은 0

    # stack → (8, 29, 2)
    x = np.stack([ch0, ch1], axis=-1)  # (8, 29, 2)
    return x

# =========================
# 6) flag 통계 / 라벨 정제
# =========================
def print_flag_stats(df: pd.DataFrame, name: str, topk=20):
    vc = df["flag"].value_counts(dropna=False).head(topk)
    print(f"\n[FLAG] {name} value_counts(top{topk})")
    print(vc)

def make_frame_labels_refined(df: pd.DataFrame, default_label: int, normal_flag_set=None):
    if (not USE_FLAG_LABELING) or (normal_flag_set is None):
        return np.full((len(df),), int(default_label), dtype=np.int32)

    flags = df["flag"].fillna("").astype(str).str.strip().to_numpy()
    y = np.empty((len(flags),), dtype=np.int32)

    for i, f in enumerate(flags):
        if f in normal_flag_set:
            y[i] = 0
        elif f == "":
            y[i] = -1
        else:
            y[i] = 1

    return y

def filter_unknown_labels(df: pd.DataFrame, y: np.ndarray):
    keep = (y != -1)
    df_out = df.loc[keep].reset_index(drop=True)
    y_out = y[keep].copy()
    removed = int((~keep).sum())
    total = int(len(y))
    removed_ratio = removed / max(1, total)
    return df_out, y_out, removed, total, removed_ratio

def print_filter_summary(name: str, y_before: np.ndarray, y_after: np.ndarray, removed: int, total: int):
    before_counts = pd.Series(y_before).value_counts().sort_index().to_dict()
    after_counts = pd.Series(y_after).value_counts().sort_index().to_dict()
    kept = len(y_after)
    ratio = removed / max(1, total)

    print(f"\n[LABEL BEFORE FILTER] {name}: {before_counts}")
    print(f"[LABEL AFTER  FILTER] {name}: {after_counts}")
    print(
        f"[FILTER SUMMARY] {name}: total={total}, kept={kept}, removed={removed}, "
        f"removed_ratio={ratio:.6f} ({ratio*100:.2f}%)"
    )

# =========================
# 7) 균등 샘플링 유틸
# =========================
def balanced_sample_attack_by_type(
    attack_df_by_name: dict,
    attack_y_by_name: dict,
    total_limit: int,
    attack_types=None,
):
    if attack_types is None:
        attack_types = sorted(list(attack_df_by_name.keys()))

    missing = [x for x in attack_types if x not in attack_df_by_name]
    if missing:
        raise RuntimeError(f"다음 공격 타입이 없습니다(확인 필요): {missing}")

    per_attack_limit = total_limit // len(attack_types)
    sampled_df = {}
    sampled_y = {}
    sampled_list_df = []
    sampled_list_y = []

    print("\n==== [BALANCED ATTACK SAMPLING] ====")
    print(f"[BALANCED] total_limit={total_limit}, num_types={len(attack_types)}, per_attack_limit={per_attack_limit}")

    for name in attack_types:
        df_i = attack_df_by_name[name].copy().reset_index(drop=True)
        y_i = attack_y_by_name[name].copy()
        before = len(df_i)

        if before > per_attack_limit:
            rng = np.random.RandomState(SEED)
            chosen_idx = rng.choice(before, size=per_attack_limit, replace=False)
            chosen_idx.sort()
            df_i = df_i.iloc[chosen_idx].reset_index(drop=True)
            y_i = y_i[chosen_idx].copy()

        after = len(df_i)
        sampled_df[name] = df_i
        sampled_y[name] = y_i
        sampled_list_df.append(df_i)
        sampled_list_y.append(y_i)

        print(
            f"[BALANCED] {name}: before={before}, after={after}, "
            f"removed={before-after}, removed_ratio={(before-after)/max(1,before):.6f} "
            f"({((before-after)/max(1,before))*100:.2f}%)"
        )
        print(f"  label dist after sampling: {pd.Series(y_i).value_counts().sort_index().to_dict()}")

    merged_df = pd.concat(sampled_list_df, axis=0).reset_index(drop=True)
    merged_y = np.concatenate(sampled_list_y, axis=0)

    print("[BALANCED] merged_attack_df shape:", merged_df.shape)
    print("[BALANCED] merged_attack_y dist:", pd.Series(merged_y).value_counts().sort_index().to_dict())

    return sampled_df, sampled_y, merged_df, merged_y, per_attack_limit

# =========================
# 8) 윈도우 생성(2채널 단일 입력)
# =========================
def build_windows_from_df_2d(df: pd.DataFrame, y_frame: np.ndarray):
    assert len(df) == len(y_frame), \
        f"df({len(df)})와 y_frame({len(y_frame)}) 길이 불일치"

    cols = ["id", "dlc"] + [f"d{i}" for i in range(8)]
    arr = df[cols].to_numpy()
    n = len(arr)

    if n < N_SEQ:
        return (
            np.zeros((0, 8, 29, 2), np.float32),
            np.zeros((0,), np.int32),
        )

    ids = np.array([id_to_int29(x) for x in arr[:, 0]], dtype=np.uint32)
    data = arr[:, 2:].astype(np.uint8)

    num_windows = n - (N_SEQ - 1)
    X_buf = np.empty((num_windows, 8, 29, 2), dtype=np.float32)
    Y_buf = np.empty((num_windows,), dtype=np.int32)
    count = 0

    for i in range(num_windows):
        window_labels = y_frame[i:i + N_SEQ]
        # 혼합 윈도우 제거: Normal+Attack 섞인 경계 구간 스킵
        if window_labels.min() != window_labels.max():
            continue
        seq_ids = ids[i:i + N_SEQ]
        last_d = data[i + N_SEQ - 1]
        X_buf[count] = mcu_preproc_one_window_2ch(seq_ids, last_d)
        # Majority Voting: 윈도우 내 과반수 라벨 사용
        Y_buf[count] = 1 if window_labels.sum() > N_SEQ // 2 else 0
        count += 1

    removed = num_windows - count
    print(f"  [WINDOW FILTER] total={num_windows}, kept={count}, removed={removed} ({removed/max(1,num_windows)*100:.2f}%)")
    return X_buf[:count], Y_buf[:count]

# =========================
# 9) split (순서 유지)
# =========================
def split_df_and_y(df: pd.DataFrame, y: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    assert len(df) == len(y), \
        f"split_df_and_y: df({len(df)})와 y({len(y)}) 길이 불일치"

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    tr = df.iloc[:n_train].reset_index(drop=True)
    va = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    te = df.iloc[n_train + n_val:].reset_index(drop=True)

    y_tr = y[:n_train].copy()
    y_va = y[n_train:n_train + n_val].copy()
    y_te = y[n_train + n_val:].copy()

    return (tr, y_tr), (va, y_va), (te, y_te)

# =========================
# 10) 모델: 2채널 단일 입력 (8x29x2)
#   ch0: F1 ID bits (7x29) zero-pad → (8,29)
#   ch1: F2 payload bits (8x8) zero-pad → (8,29)
#   Conv1: kernel(4,5) stride(2,5) → (3,5,64)
#   GlobalMaxPool → (64,) → Dense64 → Dense16 → Softmax
# =========================
def build_lcids_like_model_2d():
    in_x = tf.keras.Input(shape=(8, 29, 2), name="F1F2_2ch")

    x = tf.keras.layers.Conv2D(64, (4, 5), strides=(2, 5), padding="valid", use_bias=True)(in_x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)   # → (64,)

    x = tf.keras.layers.Dense(64, activation="relu", use_bias=True)(x)
    x = tf.keras.layers.Dense(16, activation="relu", use_bias=True)(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="p_class")(x)

    return tf.keras.Model(inputs=in_x, outputs=out)

# =========================
# 11) TFLite RAW 평가
# =========================
def tflite_predict_probs(tflite_path: str, X1: np.ndarray, X2: np.ndarray = None):
    """X1: (N, 8, 29, 2) 2채널 단일 입력"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    out_idx = out_details[0]["index"]

    N = X1.shape[0]
    probs = np.empty((N,), dtype=np.float32)

    for n in range(N):
        interpreter.set_tensor(in_details[0]["index"], X1[n:n + 1].astype(np.float32))
        interpreter.invoke()
        probs[n] = float(interpreter.get_tensor(out_idx).reshape(-1)[1])

    return probs

def confusion_2x2(y_true: np.ndarray, y_prob: np.ndarray, thr=0.5):
    y_true = y_true.astype(np.int32)
    y_pred = (y_prob >= thr).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return tn, fp, fn, tp

def metrics_from_cm(tn, fp, fn, tp):
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = (2 * prec * tpr) / max(1e-12, prec + tpr)

    return {
        "acc": acc,
        "tpr": tpr,
        "fpr": fpr,
        "precision": prec,
        "f1": f1,
    }

# =========================
# 12) segmented can_frames_dump.h 생성
# =========================
def export_can_frames_dump_h_segmented(
    normal_df: pd.DataFrame,
    attack_te_by_name: dict,
    out_path: str,
    max_frames_per_seg: int = 2500,
):
    def df_to_frames(df: pd.DataFrame, max_n: int):
        df = df.iloc[:max_n].copy().reset_index(drop=True)
        ids = df["id"].astype(str).apply(id_to_int29).to_numpy(dtype=np.uint32)
        dlc = df["dlc"].to_numpy(dtype=np.uint8)
        data = df[[f"d{i}" for i in range(8)]].to_numpy(dtype=np.uint8)
        return ids, dlc, data

    def c_array_frames(seg_name, ids, dlc, data):
        cname = seg_name.lower()
        lines = [f"static const CanFrameLite g_seg_{cname}[] = {{"]
        for i in range(len(ids)):
            b = data[i]
            lines.append(
                f"  {{ 0x{int(ids[i]):08X}u, {int(dlc[i])}u, "
                f"{{0x{b[0]:02X}u,0x{b[1]:02X}u,0x{b[2]:02X}u,0x{b[3]:02X}u,"
                f"0x{b[4]:02X}u,0x{b[5]:02X}u,0x{b[6]:02X}u,0x{b[7]:02X}u}} }},"
            )
        lines.append("};\n")
        return "\n".join(lines)

    segs = [("Normal",) + df_to_frames(normal_df, max_frames_per_seg)]
    for name in ATTACK_TYPES:
        if name not in attack_te_by_name:
            raise RuntimeError(f"attack_te_by_name에 '{name}'가 없습니다(확인 필요).")
        segs.append((name,) + df_to_frames(attack_te_by_name[name], max_frames_per_seg))

    out = []
    out.append("#ifndef CAN_FRAMES_DUMP_H")
    out.append("#define CAN_FRAMES_DUMP_H\n")
    out.append("#include <stdint.h>\n")
    out.append("typedef struct {")
    out.append("  uint32_t id29;")
    out.append("  uint8_t  dlc;")
    out.append("  uint8_t  data[8];")
    out.append("} CanFrameLite;\n")
    out.append("typedef struct {")
    out.append("  const char* name;")
    out.append("  const CanFrameLite* frames;")
    out.append("  uint32_t n;")
    out.append("} CanDumpSegment;\n")
    out.append(f"#define CAN_DUMP_NUM_SEGMENTS ({len(segs)}u)\n")
    for seg_name, ids, dlc, data in segs:
        out.append(c_array_frames(seg_name, ids, dlc, data))
    out.append("static const CanDumpSegment g_can_dump_segments[CAN_DUMP_NUM_SEGMENTS] = {")
    for seg_name, ids, dlc, data in segs:
        cname = seg_name.lower()
        out.append(
            f'  {{ "{seg_name}", g_seg_{cname}, (uint32_t)(sizeof(g_seg_{cname})/sizeof(g_seg_{cname}[0])) }},'
        )
    out.append("};\n")
    out.append("#endif\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"[DUMMY_H] saved segmented header: {out_path} (per_seg_max={max_frames_per_seg})")

# =========================
# 13) 유틸
# =========================
def concat_and_shuffle(Xa, Ya, Xb, Yb, seed=SEED):
    X = np.concatenate([Xa, Xb], axis=0)
    Y = np.concatenate([Ya, Yb], axis=0)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Y))
    return X[idx], Y[idx]

# =========================
# 14) main
# =========================
def main():
    print("==== LC-IDS MCU-based (Python) START (2D + softmax + val_acc monitor) ====")

    dataset_path = download_carhacking()
    print("[DATA] path:", dataset_path)

    csv_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True))
    txt_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.txt"), recursive=True))

    print("[DATA] csv:", len(csv_files), "txt:", len(txt_files))
    if len(csv_files) == 0:
        raise RuntimeError("공격 CSV를 찾지 못했습니다(확인 필요).")
    if len(txt_files) == 0:
        raise RuntimeError("정상 TXT를 찾지 못했습니다(확인 필요).")

    print("\n==== [CHECK] Attack CSV files ====")
    for i, f in enumerate(csv_files):
        print(f"  ({i}) {os.path.basename(f)} | path={f}")

    def quick_csv_stats(path):
        df = standardize_and_mask(read_attack_csv(path))
        n = len(df)
        flag_s = df["flag"].astype(str)
        n_empty = int(flag_s.str.strip().eq("").sum())
        return {
            "n": n,
            "n_empty_flag": n_empty,
            "empty_flag_ratio": n_empty / max(1, n),
            "bad_ts": int(pd.to_numeric(df["timestamp"], errors="coerce").isna().sum()),
            "bad_dlc": int(pd.to_numeric(df["dlc"], errors="coerce").isna().sum()),
            "flag_top10": flag_s.value_counts(dropna=False).head(10),
        }

    print("\n==== [CHECK] Per-CSV rows + flag quality ====")
    for f in csv_files:
        st = quick_csv_stats(f)
        print(f"\n[FILE] {os.path.basename(f)}")
        print(f"  rows={st['n']}, empty_flag={st['n_empty_flag']} (ratio={st['empty_flag_ratio']:.6f})")
        print(f"  bad_timestamp={st['bad_ts']} bad_dlc={st['bad_dlc']}")
        print("  flag_top10:"); print(st["flag_top10"])

    normal_txt = next((f for f in txt_files if "normal" in Path(f).name.lower()), None)
    if normal_txt is None:
        raise RuntimeError("normal txt 파일을 찾지 못했습니다(확인 필요).")
    print("[DATA] normal_txt:", normal_txt)
    normal_df = standardize_and_mask(read_normal_txt(normal_txt))

    def attack_name_from_filename(path: str) -> str:
        base = os.path.basename(path).lower()
        if "dos"   in base: return "DoS"
        if "fuzzy" in base: return "Fuzzy"
        if "rpm"   in base: return "RPM"
        if "gear"  in base: return "Gear"
        return "Unknown"

    attack_df_by_name_raw = {}
    for f in csv_files:
        name = attack_name_from_filename(f)
        df = standardize_and_mask(read_attack_csv(f))
        print(f"[LOAD PER FILE] {os.path.basename(f)} name={name} raw_rows={len(df)}")
        attack_df_by_name_raw[name] = df

    if len(normal_df) > MAX_FRAMES_NORMAL:
        normal_df = normal_df.iloc[:MAX_FRAMES_NORMAL].reset_index(drop=True)
    print("[DATA BEFORE LABEL FILTER] normal_df:", normal_df.shape)

    print_flag_stats(normal_df, "Normal")
    normal_flag_set = set(normal_df["flag"].astype(str).str.strip().unique().tolist())
    normal_flag_set.discard("")
    print("\n[FLAG] normal_flag_set:", normal_flag_set)

    y_normal_before = make_frame_labels_refined(normal_df, default_label=0, normal_flag_set=normal_flag_set)
    normal_df, y_normal_frame, rm_n, tot_n, ratio_n = filter_unknown_labels(normal_df, y_normal_before)
    print_filter_summary("Normal", y_normal_before, y_normal_frame, rm_n, tot_n)

    attack_df_by_name_refined = {}
    attack_y_by_name_refined = {}
    print("\n==== [ATTACK TYPE LABEL FILTER] ====")
    for name in ATTACK_TYPES:
        if name not in attack_df_by_name_raw:
            raise RuntimeError(f"{name} 파일이 없습니다(확인 필요).")
        df_i = attack_df_by_name_raw[name]
        y_i_before = make_frame_labels_refined(df_i, default_label=1, normal_flag_set=normal_flag_set)
        df_i_ref, y_i_ref, rm_i, tot_i, ratio_i = filter_unknown_labels(df_i, y_i_before)
        print_filter_summary(name, y_i_before, y_i_ref, rm_i, tot_i)
        attack_df_by_name_refined[name] = df_i_ref
        attack_y_by_name_refined[name] = y_i_ref

    attack_df_by_name_balanced, attack_y_by_name_balanced, attack_df, y_attack_frame, per_attack_limit = \
        balanced_sample_attack_by_type(
            attack_df_by_name=attack_df_by_name_refined,
            attack_y_by_name=attack_y_by_name_refined,
            total_limit=MAX_FRAMES_ATTACK_TOTAL,
            attack_types=ATTACK_TYPES,
        )

    print("[DATA AFTER LABEL FILTER + BALANCING] normal_df:", normal_df.shape, "attack_df:", attack_df.shape)
    print("[LABEL FINAL] normal:", pd.Series(y_normal_frame).value_counts().to_dict())
    print("[LABEL FINAL] attack:", pd.Series(y_attack_frame).value_counts().to_dict())

    (n_tr, yn_tr), (n_va, yn_va), (n_te, yn_te) = split_df_and_y(normal_df, y_normal_frame, TRAIN_RATIO, VAL_RATIO)
    (a_tr, ya_tr), (a_va, ya_va), (a_te, ya_te) = split_df_and_y(attack_df, y_attack_frame, TRAIN_RATIO, VAL_RATIO)
    print(f"[SPLIT] Normal train={len(n_tr)} val={len(n_va)} test={len(n_te)}")
    print(f"[SPLIT] Attack  train={len(a_tr)} val={len(a_va)} test={len(a_te)}")

    if EXPORT_DUMMY_SEGMENTED_H:
        attack_te_by_name = {}
        for name in ATTACK_TYPES:
            df_full = attack_df_by_name_balanced[name]
            y_full  = attack_y_by_name_balanced[name]
            (_, _), (_, _), (df_te, _) = split_df_and_y(df_full, y_full, TRAIN_RATIO, VAL_RATIO)
            attack_te_by_name[name] = df_te
        export_can_frames_dump_h_segmented(
            normal_df=n_te,
            attack_te_by_name=attack_te_by_name,
            out_path=DUMMY_SEG_H_OUT,
            max_frames_per_seg=DUMMY_PER_SEG_MAX_FRAMES,
        )

    # ── Normal+Attack 연결 윈도우 생성 ───────────────────
    # 공격 타입별로 Normal 앞 + Attack 뒤로 이어붙여서 윈도우 생성
    # → 실제 CAN 버스처럼 Normal→Attack 전환 경계 구간 포함
    # → 혼합 윈도우 제거가 실질적으로 동작함
    print("[BUILD] windows (2채널, Normal+Attack 연결) ...")

    def build_windows_concat(n_df, yn, a_df, ya, seed):
        # Normal 뒤에 Attack을 이어 붙임
        combined_df = pd.concat([n_df, a_df], axis=0).reset_index(drop=True)
        combined_y  = np.concatenate([yn, ya], axis=0)
        X, Y = build_windows_from_df_2d(combined_df, combined_y)
        # shuffle
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Y))
        return X[idx], Y[idx]

    X_tr, Y_tr = build_windows_concat(n_tr, yn_tr, a_tr, ya_tr, seed=SEED)
    X_va, Y_va = build_windows_concat(n_va, yn_va, a_va, ya_va, seed=SEED+1)
    X_te, Y_te = build_windows_concat(n_te, yn_te, a_te, ya_te, seed=SEED+2)

    gc.collect()

    print("[SHAPE] train:", X_tr.shape, Y_tr.shape)
    print("[SHAPE] val  :", X_va.shape, Y_va.shape)
    print("[SHAPE] test :", X_te.shape, Y_te.shape)

    # ── 기존 캐시 삭제 ───────────────────────────────────
    SAVE_PATH = "lcids_2ch_windows.npz"
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print(f"[CACHE] 기존 캐시 삭제: {SAVE_PATH}")

    # ── 윈도우 저장 후 종료 ──────────────────────────────
    print(f"[SAVE] 저장 중: {SAVE_PATH} ...")
    np.savez_compressed(
        SAVE_PATH,
        X_tr=X_tr, Y_tr=Y_tr,
        X_va=X_va, Y_va=Y_va,
        X_te=X_te, Y_te=Y_te,
    )
    size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"[SAVE] 완료  {size_mb:.1f} MB  →  {SAVE_PATH}")
    print("[SAVE] 이제 lcids_train_only.py 로 학습하세요.")


if __name__ == "__main__":
    main()