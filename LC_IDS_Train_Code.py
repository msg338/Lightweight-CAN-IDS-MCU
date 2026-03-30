#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LC-IDS 학습 전용 v2
- Warmup + Cosine Decay LR 스케줄
- Label Smoothing
- lcids_2ch_windows.npz 로드 후 바로 학습
"""

import os
import gc
import math
import numpy as np
import tensorflow as tf

# =========================
# 설정
# =========================
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOAD_PATH  = "lcids_2ch_windows.npz"
OUT_H5     = "lcids_mcu_fp32_2d_last.h5"
BEST_H5    = "lcids_mcu_fp32_2d_best.h5"
OUT_TFLITE = "lcids_mcu_fp32_2d_best.tflite"

BATCH_SIZE     = 512
EPOCHS         = 60
WARMUP_EPOCHS  = 3       # Warmup 구간 (LR 0 → LR_MAX)
LR_MAX         = 1e-3    # Cosine decay 최대 LR
LR_MIN         = 1e-6    # Cosine decay 최소 LR
LABEL_SMOOTH   = 0.0     # Label smoothing (0이면 비활성)

EARLY_STOP_PATIENCE = 10
MONITOR_METRIC      = "val_acc"
MONITOR_MODE        = "max"

# =========================
# Warmup + Cosine Decay 스케줄러
# =========================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_max      = lr_max
        self.lr_min      = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.warmup_steps
        total  = self.total_steps

        # Warmup 구간: 선형 증가
        warmup_lr = self.lr_max * (step / tf.cast(warmup, tf.float32))

        # Cosine decay 구간
        progress  = (step - warmup) / tf.cast(total - warmup, tf.float32)
        progress  = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + tf.math.cos(math.pi * progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(
            lr_max=self.lr_max, lr_min=self.lr_min,
            warmup_steps=self.warmup_steps, total_steps=self.total_steps,
        )

# =========================
# 모델 정의 (여기만 바꾸면 됨)
# =========================
def build_model():
    in_x = tf.keras.Input(shape=(8, 29, 2), name="F1F2_2ch")
    x = tf.keras.layers.Conv2D(32, (4, 5), strides=(2, 5), padding="valid", use_bias=True)(in_x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu", use_bias=True)(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="p_class")(x)
    return tf.keras.Model(inputs=in_x, outputs=out)

# =========================
# TFLite 평가
# =========================
def tflite_predict_probs(path, X):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    probs = np.empty((len(X),), dtype=np.float32)
    for i in range(len(X)):
        interp.set_tensor(in_idx, X[i:i+1].astype(np.float32))
        interp.invoke()
        probs[i] = float(interp.get_tensor(out_idx).reshape(-1)[1])
    return probs

def confusion_2x2(y_true, y_prob, thr=0.5):
    y_true = y_true.astype(np.int32)
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    return tn, fp, fn, tp

def metrics_from_cm(tn, fp, fn, tp):
    acc  = (tp+tn) / max(1, tp+tn+fp+fn)
    tpr  = tp / max(1, tp+fn)
    fpr  = fp / max(1, fp+tn)
    prec = tp / max(1, tp+fp)
    f1   = 2*prec*tpr / max(1e-12, prec+tpr)
    return dict(acc=acc, tpr=tpr, fpr=fpr, precision=prec, f1=f1)

# =========================
# main
# =========================
def main():
    # 1) 데이터 로드
    if not os.path.exists(LOAD_PATH):
        raise RuntimeError(f"캐시 없음: {LOAD_PATH}\n먼저 lcids_save_windows.py 실행하세요.")
    print(f"[LOAD] {LOAD_PATH} ...")
    cache = np.load(LOAD_PATH)
    X_tr, Y_tr = cache["X_tr"], cache["Y_tr"]
    X_va, Y_va = cache["X_va"], cache["Y_va"]
    X_te, Y_te = cache["X_te"], cache["Y_te"]
    print(f"[LOAD] 완료  train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")

    # 2) LR 스케줄 계산
    steps_per_epoch = math.ceil(len(X_tr) / BATCH_SIZE)
    total_steps     = EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
    print(f"[LR] steps_per_epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")
    print(f"[LR] LR_MAX={LR_MAX}  LR_MIN={LR_MIN}  WARMUP_EPOCHS={WARMUP_EPOCHS}")
    print(f"[LABEL] label_smoothing={LABEL_SMOOTH}")

    lr_schedule = WarmupCosineDecay(
        lr_max=LR_MAX, lr_min=LR_MIN,
        warmup_steps=warmup_steps, total_steps=total_steps,
    )

    # 3) Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_va, Y_va)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 4) 모델 학습
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            ignore_class=None,
        ),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()

    # Label Smoothing은 custom loss로 적용
    if LABEL_SMOOTH > 0:
        n_classes = 2
        @tf.function
        def smoothed_loss(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_onehot = tf.one_hot(y_true, n_classes)
            y_smooth = y_onehot * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / n_classes
            return tf.reduce_mean(
                -tf.reduce_sum(y_smooth * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1)
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            loss=smoothed_loss,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC, patience=EARLY_STOP_PATIENCE,
            verbose=1, mode=MONITOR_MODE, restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_H5, monitor=MONITOR_METRIC,
            save_best_only=True, verbose=1, mode=MONITOR_MODE,
        ),
    ]

    print("\n==== 학습 시작 ====")
    print(f"Warmup {WARMUP_EPOCHS} epochs → Cosine Decay → EarlyStopping(patience={EARLY_STOP_PATIENCE})")
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
    model.save(OUT_H5)
    print("[SAVED]", OUT_H5)

    # 5) TFLite 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(OUT_TFLITE, "wb") as f:
        f.write(converter.convert())
    size_kb = os.path.getsize(OUT_TFLITE) / 1024
    print(f"[SAVED] {OUT_TFLITE}  ({size_kb:.1f} KB)")

    # 6) 평가
    print("\n==== TFLite evaluation ====")
    probs_tfl = tflite_predict_probs(OUT_TFLITE, X_te)
    tn, fp, fn, tp = confusion_2x2(Y_te, probs_tfl)
    print("[TFLite]", {k: f"{v:.4f}" for k, v in metrics_from_cm(tn, fp, fn, tp).items()})
    print("[CM]", np.array([[tn, fp], [fn, tp]], dtype=np.int64))

    print("\n==== Keras BEST ====")
    probs = model.predict(X_te, batch_size=BATCH_SIZE, verbose=0)[..., 1].astype(np.float32)
    tn2, fp2, fn2, tp2 = confusion_2x2(Y_te, probs)
    m2 = metrics_from_cm(tn2, fp2, fn2, tp2)
    print("[Keras]", {k: f"{v:.4f}" for k, v in m2.items()})
    print(f"\n==== DONE  acc={m2['acc']:.4f}  f1={m2['f1']:.4f} ====")

    del X_tr, Y_tr, X_va, Y_va, X_te, Y_te
    gc.collect()

if __name__ == "__main__":
    main()