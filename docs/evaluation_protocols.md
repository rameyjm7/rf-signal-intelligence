# Evaluation Protocols

This page summarizes the evaluation protocols used across the repo so result rows are not mistaken for directly comparable claims.

## Protocol A: Full-Distribution All-SNR Test

| Field | Value |
|---|---|
| Dataset | RML2016, RML2018, or DeepRadar2022 |
| Sample count | Dataset/notebook dependent |
| SNR range | Full available SNR range |
| Class balance | Dataset dependent |
| Train/test split | Notebook/config specific |
| Checkpoint used | Best available model for that notebook run |
| Label mapping rule | Dataset-native class order unless noted |
| Metric | Accuracy, macro F1, weighted F1 |
| Comparable to other rows | Only comparable to rows with the same dataset, split, SNR range, and label mapping |

## Protocol B: High-SNR-Only Test

| Field | Value |
|---|---|
| Dataset | Usually RML2016 or NoisyDroneRFv2 |
| Sample count | Filtered by SNR threshold |
| SNR range | Thresholded, such as SNR > -2 dB or SNR >= 20 dB |
| Class balance | May be natural or balanced |
| Train/test split | Same base split as source evaluation |
| Checkpoint used | Same checkpoint as full test unless noted |
| Label mapping rule | Dataset-native class order unless noted |
| Metric | Accuracy, macro F1, weighted F1 |
| Comparable to other rows | Not directly comparable to all-SNR rows |

## Protocol C: Class-Balanced High-SNR Slice

| Field | Value |
|---|---|
| Dataset | RML2018 or NoisyDroneRFv2 |
| Sample count | Equal samples per class |
| SNR range | High-SNR slice |
| Class balance | Balanced |
| Train/test split | Derived from held-out or notebook-selected slice |
| Checkpoint used | Notebook/config specific |
| Label mapping rule | May require explicit class-order calibration |
| Metric | Accuracy, macro F1, weighted F1 |
| Comparable to other rows | Only comparable to similarly balanced high-SNR slices |

## Protocol D: Cross-Dataset Ensemble Sample

| Field | Value |
|---|---|
| Dataset | Sampled combination of supported datasets |
| Sample count | Notebook/config specific |
| SNR range | Mixed |
| Class balance | Mixed |
| Train/test split | Evaluation-only sampled set |
| Checkpoint used | Pinned best checkpoints per dataset |
| Label mapping rule | Calibrated per dataset/model |
| Metric | Accuracy and confusion matrix |
| Comparable to other rows | No; this is a system integration check |

## Protocol E: Noisy Drone Held-Out Natural Split

| Field | Value |
|---|---|
| Dataset | NoisyDroneRFv2 |
| Sample count | 649 in current result snapshot |
| SNR range | Filtered with `NOISY_DRONE_MIN_SNR_DB=-6` |
| Class balance | Natural held-out distribution |
| Train/test split | Stratified 80/20 split |
| Checkpoint used | `noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras` |
| Label mapping rule | NoisyDroneRFv2 class mapping from manifest/model labels |
| Metric | Accuracy, macro F1, weighted F1 |
| Comparable to other rows | Comparable to repeated natural-split NoisyDroneRFv2 runs with the same filters |

## Protocol F: Noisy Drone Held-Out Balanced Split

| Field | Value |
|---|---|
| Dataset | NoisyDroneRFv2 |
| Sample count | 203 in current result snapshot |
| SNR range | Filtered with `NOISY_DRONE_MIN_SNR_DB=-6` |
| Class balance | 29 held-out windows per class |
| Train/test split | Balanced subset of held-out test split |
| Checkpoint used | `noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras` |
| Label mapping rule | NoisyDroneRFv2 class mapping from manifest/model labels |
| Metric | Accuracy, macro F1, weighted F1 |
| Comparable to other rows | Comparable to balanced NoisyDroneRFv2 rows with the same split and filters |

## Protocol G: Live OTA SDR-To-SDR Class Sweep

| Field | Value |
|---|---|
| Dataset | NoisyDroneRFv2 IQ samples replayed over SDR |
| Sample count | 70 trials in current SNR >= 20 dB sweep |
| SNR range | Source files selected with `--tx-min-snr 20` |
| Class balance | 10 trials per class across seven classes |
| Train/test split | Uses held-out/source dataset files selected by sweep command |
| Checkpoint used | NoisyDroneRFv2 VGG full-complex spectrogram model |
| Label mapping rule | Model labels plus target class from dataset filename/manifest |
| Metric | Exact final prediction match, confusion matrix, per-trial confidence |
| Comparable to other rows | No; this is controlled OTA replay/receive evidence, not offline dataset evaluation |
