# Lightweight-CAN-IDS-MCU

<img width="697" height="423" alt="image" src="https://github.com/user-attachments/assets/46c1425e-9084-4c79-b38a-86037088969f" />

<img width="718" height="299" alt="image" src="https://github.com/user-attachments/assets/c4738f16-a258-4629-ad58-55bab8146ed5" />

<img width="895" height="181" alt="image" src="https://github.com/user-attachments/assets/7be8c5a6-5a07-4d10-97ae-72c47127a773" />

## Overview
Lightweight 2D CNN-based CAN Bus Intrusion Detection System
deployed on STM32F401RE (MCU).

## Motivation
Based on LC-IDS (Im & Lee, 2025), which uses dual-input 
(Feature 1 + Feature 2). However, dual-input increases 
inference latency on resource-constrained MCUs.

## Design Decision Process

### Step 1. Dual-input → Single dual-channel input
Original LC-IDS uses separate dual-input (Feature 1 + Feature 2).
Dual-input increases inference latency on MCU.
→ Merged into single 8×29×2 dual-channel tensor.

### Step 2. Concat structure → GlobalMaxPooling
MaxPool + Concat architecture increases parameters unnecessarily.
Tried GlobalMaxPooling + Concat but still suboptimal.
→ Final decision: Single input + GlobalMaxPooling2D
  - Fewer parameters
  - Lower latency
  - Better efficiency on STM32F401RE

## Model Architecture
- Input: 8×29×2 (ID bits + Payload bits, single tensor)
- Conv2D(32, (4,5), stride(2,5)) → GlobalMaxPooling2D
- Dense(32) → Softmax
- Parameters: 2,434

## Results
- Accuracy: 99.2%
- Inference latency: 4ms (STM32F401RE)
- Attack types: DoS, Fuzzy, Gear, RPM

## Dataset
Kaggle Car-Hacking Dataset

## References
- Im, H., & Lee, S. (2025). TinyML-Based Intrusion Detection 
  System for In-Vehicle Network. *IEEE Embedded Systems Letters*, 17(2), 67.
