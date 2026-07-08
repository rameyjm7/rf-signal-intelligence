# Offline Model Results

This page keeps the detailed offline dataset results out of the top-level README while preserving the technical trail for reviewers.

For the live SDR-to-SDR NoisyDroneRFv2 evidence, see:

- [70-trial OTA SDR-to-SDR sweep report](../../results/noisy_drone_rf_v2/snr20_class_sweep_results.md)
- [NoisyDroneRFv2 result card](noisy_drone_rf_v2/README.md)
- [Jetson TensorRT benchmark/profile summary](../../results/benchmarks/noisy_drone_tensorrt_jetson.md)

## Latest Comparison Snapshot

| Dataset | Model | Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---|---|---:|---:|---:|---:|
| Noisy Drone RF v2 | VGG full-complex spectrogram | Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| Noisy Drone RF v2 | VGG full-complex spectrogram | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| RML2016 | CNN-transformer | All SNR levels | 0.6645 | 0.6592 | 0.6592 | 44,000 |
| RML2016 | CNN-transformer | SNR > -2 dB | 0.8969 | 0.8900 | 0.8913 | n/a |
| RML2018 | LSTM continued checkpoint | Current notebook protocol | 0.8295 | n/a | n/a | n/a |
| DeepRadar2022 | CNN-transformer continued stage 2 | Current notebook protocol | 0.4461 | 0.3973 | 0.3973 | n/a |

## Noisy Drone RF v2

Canonical model:

```text
models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras
```

Pipeline flow:

- `33_vgg_spectrogram_noisy_drone_rf_v2.py`: training/evaluation experiment and baseline artifacts.
- `44_evaluation_noisy_drone_rf_v2.py`: eval-only pipeline for the canonical VGG model.
- `50_evaluation_comparison.py`: consolidated comparison with a dedicated Noisy Drone RF v2 eval-only cell.

Current metrics from pipelines `33`, `44`, and `50` agree on the same held-out split configuration: `NOISY_DRONE_MIN_SNR_DB=-6`, `NOISY_DRONE_DATA_FRACTION=0.25`, one eval window.

| Source | Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---|---:|---:|---:|---:|
| `33` | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| `44` | Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| `44` | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| `50` | Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| `50` | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |

Saved result artifacts:

- [Live OTA SDR-to-SDR class sweep report](../../results/noisy_drone_rf_v2/class_sweep_results.md)
- [70-trial OTA SDR-to-SDR SNR >= 20 dB sweep report](../../results/noisy_drone_rf_v2/snr20_class_sweep_results.md)
- `outputs/noisy_drone_rf_v2_eval/33_noisy_drone_rf_v2_vgg_full_complex_spectrogram_balanced_confusion_matrix.png`
- `outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_balanced_confusion_matrix.png`
- `outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr.png`
- `outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr_per_class.png`
- `outputs/50_evaluation_comparison/50_cross_dataset_model_comparison.csv`

### Balanced Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| DJI | 1.00 | 0.93 | 0.96 | 29 |
| FutabaT14 | 1.00 | 0.97 | 0.98 | 29 |
| FutabaT7 | 1.00 | 1.00 | 1.00 | 29 |
| Graupner | 1.00 | 1.00 | 1.00 | 29 |
| Noise | 0.88 | 1.00 | 0.94 | 29 |
| Taranis | 1.00 | 0.97 | 0.98 | 29 |
| Turnigy | 1.00 | 1.00 | 1.00 | 29 |
| **Accuracy** |  |  | **0.98** | **203** |
| **Macro avg** | **0.98** | **0.98** | **0.98** | **203** |
| **Weighted avg** | **0.98** | **0.98** | **0.98** | **203** |

## RML2016

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| All SNR levels | 0.6645 | 0.6592 | 0.6592 | 44,000 |
| SNR > -2 dB | 0.8969 | 0.8900 | 0.8913 | n/a |
| SNR > 5 dB legacy snapshot | 0.93 | 0.92 | 0.92 | 15,332 |

Legacy confusion matrix:

![RML2016 Confusion Matrix](https://github.com/user-attachments/assets/6eebbb20-105d-4c9c-ba17-7f2ec11e070f)

## RML2018

Recent local snapshot:

| Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| Current notebook protocol | 0.8295 | n/a | n/a | n/a |
| Highest-SNR class-balanced slice with class-order calibration | 0.9465 | 0.94 | 0.94 | 4,800 |

Legacy full-distribution baseline was approximately 72% across 72,000 evaluation samples. Protocols differ, so these rows should not be treated as directly comparable without checking split, SNR range, and label mapping.

Legacy figures:

![RML2018 Image 1](https://github.com/user-attachments/assets/e23bbd81-9f4f-4d7a-9bd4-11e0a3625044)
![RML2018 Image 2](https://github.com/user-attachments/assets/99d3b667-93d4-430e-abf3-aa6b4c743a31)
![RML2018 Image 3](https://github.com/user-attachments/assets/5e8d2c18-5c62-489e-84ea-8b2648eca610)

## DeepRadar2022

| Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| Current comparison snapshot | 0.4461 | 0.3973 | 0.3973 | n/a |
| Legacy all-SNR local snapshot | 0.8433 | 0.84 | 0.84 | 156,400 |

The DeepRadar rows are retained for continuity, but they reflect different protocol states and should be interpreted with the evaluation notes.

Legacy figures:

![DeepRadar Image 1](https://github.com/user-attachments/assets/a2e6d2dc-ef18-4bdd-a8c8-31d2bbb77a0f)
![DeepRadar Image 2](https://github.com/user-attachments/assets/68843377-7fc4-45ba-9f16-9a40b9ecc2c9)
![DeepRadar Image 3](https://github.com/user-attachments/assets/0754f4da-e8d6-4cd0-9627-056e932a2865)
![DeepRadar Image 4](https://github.com/user-attachments/assets/c9eb6c5b-737f-4273-ba68-0ac3d13e3aab)

## Cross-Dataset Ensemble

Recent cross-dataset ensemble evaluation is aligned with pinned best-checkpoint loading and class-order calibration. A recent sampled combined run produced approximately 0.96 overall accuracy.

Legacy figure:

<img width="2184" height="1990" alt="Cross-dataset ensemble confusion matrix" src="https://github.com/user-attachments/assets/371e4354-fa85-4b50-9143-50fbcbdb7927" />
