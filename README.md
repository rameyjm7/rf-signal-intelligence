# RF Signal Intelligence

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![Docker](https://img.shields.io/badge/Docker-GPU--Ready-blue)

**Maintainer:** Jacob M. Ramey  
LinkedIn: https://www.linkedin.com/in/rameyjm/

GPU-ready RF/IQ machine-learning workspace for modulation recognition, radar waveform analysis, signal preprocessing, model evaluation, and reproducible dataset workflows.

The current work centers on wireless signal classification, with room to grow into broader RFML tasks such as signal embeddings, anomaly detection, streaming inference, and SDR-backed dataset generation.

## Table of Contents
- [Overview](#overview)
- [Current Progress (July 2026)](#current-progress-july-2026)
- [Repository Layout](#repository-layout)
- [Datasets](#datasets)
- [Results: RML2016](#results-rml2016)
- [Results: RML2018](#results-rml2018)
- [Results: DeepRadar2022](#results-deepradar2022)
- [Results: Noisy Drone RF v2](#results-noisy-drone-rf-v2)
- [Requirements](#requirements)
- [Local Setup](#local-setup)
- [CLI Usage](#cli-usage)
- [Docker](#docker)
- [Notes](#notes)
- [Citation](#citation)

## Overview

This repository contains training pipelines, evaluation notebooks, saved models, and documentation for deep-learning-based RF signal intelligence.

Implemented workflows include:
- LSTM and BiLSTM architectures for raw I/Q modeling
- CNN + recurrent hybrids for time-frequency structure
- Cross-dataset experimentation on RML2016, RML2018, and DeepRadar2022
- GPU-ready execution via Docker/Apptainer

## Current Progress (July 2026)

- RML2018 was rebaselined with a new continuation training run; best checkpoint evaluation reached `0.8295` accuracy on the current filtered protocol used in notebook evaluation.
- Cross-dataset ensemble evaluation (`43`) is now aligned with pinned best-checkpoint loading and class-order calibration, producing stable combined results (recent run: `0.96` overall on the sampled combined set).
- Notebook naming and pipeline flow were standardized:
  - training: `30_lstm_rml2016.ipynb`, `31_lstm_rml2018.ipynb`, `32_lstm_deepradar2022.ipynb`
  - evaluation: `40_evaluation_rml2016.ipynb`, `41_evaluation_rml2018.ipynb`, `42_evaluation_deepradar2022.ipynb`, `43_evaluation_cross_dataset_ensemble.ipynb`, `50_evaluation_comparison.ipynb`
- Evaluation notebooks now save local artifacts under `outputs/` (confusion matrices, classification reports, SNR charts, and training-curve plots) so notebook outputs can be cleared while retaining reproducible figures/tables.
- Artifact review notebook added: `notebooks/99_outputs_artifact_review.ipynb` for consolidating saved outputs before README/report updates.

### Latest Notebook 50 Run (Local)

| Dataset | Model | Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---|---|---:|---:|---:|---:|
| Noisy Drone RF v2 | VGG full-complex spectrogram | Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| Noisy Drone RF v2 | VGG full-complex spectrogram | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| RML2016 | CNN-transformer | All SNR levels | 0.6645 | 0.6592 | 0.6592 | 44,000 |
| RML2016 | CNN-transformer | SNR > -2 dB | 0.8969 | 0.8900 | 0.8913 | n/a |
| RML2018 | LSTM continued checkpoint | All test | 0.8295 | n/a | n/a | n/a |
| DeepRadar2022 | CNN-transformer continued stage 2 | All test | 0.4461 | 0.3973 | 0.3973 | n/a |

Notebook `50` now evaluates Noisy Drone RF v2 in a dedicated eval-only cell and writes consolidated comparison artifacts under `outputs/50_evaluation_comparison/`.

## Repository Layout

```text
src/rf_signal_intelligence/   Python package
  core/                           Maintained runtime/training components
  data/                           Dataset manifests and IQ loading helpers
  features/                       Reusable RF feature extraction
  workflows/                      Config-driven training/evaluation/export workflows
  models/                         Maintained model definitions
  legacy/                         Compatibility shims for older experiments
  base/                           Backward-compatible import wrappers
configs/                          Dataset and model registries (YAML)
data/                             Datasets (RML2016, RML2018, DeepRadar2022)
models/                           Saved model artifacts
outputs/                          Generated runtime outputs (stats, logs, new artifacts)
notebooks/                        Reproducible notebooks
docker/                           Docker and Apptainer build/runtime files
docs/                             Project reports and papers
tests/                            Test and integration checks
```

Archived experiments and prototype notebooks were moved out of this working branch and are
preserved on the `archive/legacy-notebooks` branch.

## Datasets

### RML2016.10A
11 modulation types, SNR from -20 dB to +18 dB.

### RML2018.01A
24 classes; used for larger-scale training/evaluation and cross-dataset checks.

### DeepRadar2022
Radar waveform dataset used for CNN-BiLSTM style modeling and transfer evaluation.

Detailed dataset cards are available under [`docs/dataset_cards/`](docs/dataset_cards/):

- [Noisy Drone RF v2](docs/dataset_cards/noisy_drone_rf_v2.md)
- [RML2016.10A](docs/dataset_cards/rml2016.md)
- [RML2018.01A](docs/dataset_cards/rml2018.md)
- [DeepRadar2022](docs/dataset_cards/deepradar2022.md)

Model cards are available under [`docs/model_cards/`](docs/model_cards/):

- [NoisyDroneRFv2 VGG](docs/model_cards/noisy_drone_rf_v2_vgg.md)
- [RML2016 CNN-transformer](docs/model_cards/rml2016_cnn_transformer.md)
- [RML2018 LSTM](docs/model_cards/rml2018_lstm.md)
- [DeepRadar2022 CNN-transformer](docs/model_cards/deepradar2022_cnn_transformer.md)

NVIDIA edge deployment path:

- [Jetson TensorRT deployment guide](docs/jetson_tensorrt_deployment.md)

## Results: RML2016

### Summary
- Accuracy (all SNR): 67.0%
- Accuracy (SNR > 5 dB): 94%
- Macro F1: 0.68
- Weighted F1: 0.68

### Current Local Snapshot (Notebook 50)

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| All SNR levels | 0.67 | 0.69 | 0.69 | 44,000 |
| SNR > 5 dB | 0.93 | 0.92 | 0.92 | 15,332 |

### Confusion Matrix
![RML2016 Confusion Matrix](https://github.com/user-attachments/assets/6eebbb20-105d-4c9c-ba17-7f2ec11e070f)

## Results: RML2018

### Overall Accuracy
Legacy baseline (full-distribution run): approx. 72% across 72,000 evaluation samples.

### Results Figure 1
![RML2018 Image 1](https://github.com/user-attachments/assets/e23bbd81-9f4f-4d7a-9bd4-11e0a3625044)

### Classification Report (All SNRs)

```text
precision    recall  f1-score   support

128APSK 0.36 0.16 0.23 3000
128QAM  0.42 0.49 0.45 3000
16APSK  0.91 0.90 0.90 3000
16PSK   0.86 0.76 0.81 3000
16QAM   0.75 0.92 0.83 3000
256QAM  0.92 0.74 0.82 3000
32APSK  0.87 0.80 0.83 3000
32PSK   0.98 0.97 0.98 3000
32QAM   0.91 0.90 0.90 3000
4ASK    0.63 0.56 0.60 3000
64APSK  0.45 0.69 0.54 3000
64QAM   0.60 0.84 0.70 3000
8ASK    0.46 0.83 0.60 3000
8PSK    0.63 0.90 0.74 3000
AM-DSB-SC 0.39 0.15 0.22 3000
AM-DSB-WC 1.00 1.00 1.00 3000
AM-SSB-SC 0.66 0.60 0.63 3000
AM-SSB-WC 0.71 0.45 0.55 3000
BPSK    0.78 0.85 0.81 3000
FM      0.94 0.98 0.96 3000
GMSK    0.88 0.87 0.87 3000
OOK     0.84 0.96 0.90 3000
OQPSK   0.29 0.04 0.06 3000
QPSK    0.78 0.94 0.85 3000

accuracy 0.72 72000
```

### Additional RML2018 Figures

![RML2018 Image 2](https://github.com/user-attachments/assets/99d3b667-93d4-430e-abf3-aa6b4c743a31)
![RML2018 Image 3](https://github.com/user-attachments/assets/5e8d2c18-5c62-489e-84ea-8b2648eca610)

### Current Local Snapshot (Notebook 50)

| Eval Protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| Highest-SNR class-balanced slice with class-order calibration | 0.9465 | 0.94 | 0.94 | 4,800 |

Notes:
- Mapping calibration in notebook `50` selected `LabelEncoder` order (`acc_orig=0.0192`, `acc_fixed=0.0994`, `acc_le=0.9465`).
- This snapshot is not directly comparable to full-distribution all-SNR evaluations.

## Results: DeepRadar2022

### CNN-BiLSTM Hybrid Evaluation

![DeepRadar Image 1](https://github.com/user-attachments/assets/a2e6d2dc-ef18-4bdd-a8c8-31d2bbb77a0f)
![DeepRadar Image 2](https://github.com/user-attachments/assets/68843377-7fc4-45ba-9f16-9a40b9ecc2c9)
![DeepRadar Image 3](https://github.com/user-attachments/assets/0754f4da-e8d6-4cd0-9627-056e932a2865)
![DeepRadar Image 4](https://github.com/user-attachments/assets/c9eb6c5b-737f-4273-ba68-0ac3d13e3aab)

### Current Local Snapshot (Notebook 50)

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| All SNR levels | 0.8433 | 0.84 | 0.84 | 156,400 |

## Results: Cross-Dataset Ensemble

<img width="2184" height="1990" alt="image" src="https://github.com/user-attachments/assets/371e4354-fa85-4b50-9143-50fbcbdb7927" />

## Results: Noisy Drone RF v2

The current canonical Noisy Drone RF v2 model is the VGG full-complex spectrogram model saved at:

`models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras`

Notebook flow:
- `33_vgg_spectrogram_noisy_drone_rf_v2.ipynb`: training/evaluation experiment and baseline saved artifacts.
- `44_evaluation_noisy_drone_rf_v2.ipynb`: eval-only notebook for the canonical VGG model.
- `50_evaluation_comparison.ipynb`: final comparison, with a dedicated Noisy Drone RF v2 eval-only cell.

Current metrics from notebooks `33`, `44`, and `50` agree on the same held-out split configuration (`NOISY_DRONE_MIN_SNR_DB=-6`, `NOISY_DRONE_DATA_FRACTION=0.25`, one eval window).

| Source | Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---|---:|---:|---:|---:|
| `33` | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| `44` | Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| `44` | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| `50` | Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| `50` | Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |

Saved result artifacts:
- [Live OTA SDR-to-SDR class sweep report](results/noisy_drone_rf_v2/class_sweep_results.md)
- [70-trial OTA SDR-to-SDR SNR >= 20 dB sweep report](results/noisy_drone_rf_v2/snr20_class_sweep_results.md)
- [33 balanced confusion matrix](outputs/noisy_drone_rf_v2_eval/33_noisy_drone_rf_v2_vgg_full_complex_spectrogram_balanced_confusion_matrix.png)
- [44 balanced confusion matrix](outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_balanced_confusion_matrix.png)
- [44 accuracy vs. SNR](outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr.png)
- [44 per-class accuracy vs. SNR](outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr_per_class.png)
- [50 cross-dataset comparison](outputs/50_evaluation_comparison/50_cross_dataset_model_comparison.csv)

Existing result snapshots are retained below for continuity.

<img width="1093" height="989" alt="image" src="https://github.com/user-attachments/assets/b38df917-d669-472b-bdbb-6c0df3673898" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/ce206ad4-3df6-4a9c-8665-674df6181c6f" />

### VGG Full-Complex Spectrogram Balanced Report

**Accuracy:** 0.9803

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


## Live RF Drone Classifier

Run the NoisyDroneRFv2 model against IQ playback, live SDR receive, or SDR-to-SDR over-the-air replay.

Pipeline:

```text
IQ source -> windowing -> preprocessing/spectrogram -> model inference -> class/confidence -> latency/throughput reporting
```

The live script accepts `.npy`, `.npz`, `.pt`, and raw complex64 `.bin` / `.c64` IQ files for playback. It also supports SoapySDR receive and an optional TX path for replaying labeled NoisyDroneRFv2 samples from one SDR into another.

IQ file playback:

```bash
python scripts/live_noisy_drone_rf_classifier.py \
  --iq-file outputs/rx_debug.npy \
  --model models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras \
  --window-samples 1048576 \
  --nfft 1024 \
  --hop 1024 \
  --time-bins 1024 \
  --once
```

Live SDR receive with SoapySDR:

```bash
python scripts/live_noisy_drone_rf_classifier.py \
  --device-args driver=hackrf \
  --freq 2.399e9 \
  --sample-rate 20e6 \
  --bandwidth 20e6 \
  --gain 60 \
  --model models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras
```

SDR-to-SDR replay and receive demo:

```bash
python scripts/live_noisy_drone_rf_classifier.py \
  --tx \
  --tx-dataset-dir /data/rameyjm7/datasets/NoisyDroneRFv2 \
  --tx-class-name DJI \
  --tx-min-snr 24 \
  --device-args driver=hackrf \
  --tx-device-args driver=bladerf \
  --freq 2.399e9 \
  --sample-rate 20e6 \
  --bandwidth 20e6 \
  --tx-bandwidth 20e6 \
  --gain 60 \
  --tx-gain 60 \
  --once \
  --save-rx-iq outputs/rx_debug.npy
```

Reproduce the documented OTA class-sweep report:

```bash
python scripts/live_noisy_drone_rf_classifier.py \
  --tx-test-all-classes \
  --tx-test-classes DJI,FutabaT14,FutabaT7,Graupner,Noise,Taranis,Turnigy \
  --tx-test-count 10 \
  --tx-min-snr 20 \
  --tx-test-output-csv outputs/noisy_drone_rf_v2_snr20_class_sweep.csv \
  --tx-test-output-md results/noisy_drone_rf_v2/snr20_class_sweep_results.md \
  --tx-test-save-rx-dir outputs/noisy_drone_rf_v2_snr20_iq \
  --tx-test-save-plots-dir results/noisy_drone_rf_v2/snr20_waterfalls
```

Suggested resume bullet:

```text
Built a live RF drone-classification pipeline with IQ playback/receive, windowed preprocessing, spectrogram-based deep-learning inference, confidence reporting, and latency instrumentation for real-time RF sensor-processing workflows.
```


## Requirements

- Python 3.10+
- `pip`
- Optional: NVIDIA GPU stack for accelerated TensorFlow runs

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For contributor tooling (lint, tests, hooks):

```bash
pip install -e ".[dev,test]"
pre-commit install
```

For NoisyDroneRFv2 `.pt` dataset evaluation:

```bash
pip install -e ".[noisy-drone]"
```

For GPU-focused environments (Linux):

```bash
pip install -e ".[gpu]"
```

## Data

By default, the CLI searches for:
- `data/RML2016/RML2016.10a_dict.pkl`
- `RML2016.10a_dict.pkl` (repository root)

If your dataset is elsewhere, pass `--data-path`.

## CLI Usage

The preferred reproducible workflow is moving from notebook-only execution to the
`rfsi` CLI plus small notebooks that call reusable code under `src/`.

```bash
# Train or continue the canonical NoisyDroneRFv2 VGG spectrogram model.
rfsi train --config configs/noisy_drone_vgg.yaml

# Evaluate the canonical NoisyDroneRFv2 VGG spectrogram model.
rfsi evaluate \
  --config configs/noisy_drone_vgg.yaml \
  --checkpoint models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras

# Rebuild the cross-dataset comparison artifacts.
rfsi compare --config configs/evaluation_comparison.yaml

# Export the NoisyDroneRFv2 model for deployment work.
rfsi export-onnx \
  --config configs/noisy_drone_vgg.yaml \
  --out models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample-out models/noisy_drone_rf_v2/sample_input.npy \
  --labels-out models/noisy_drone_rf_v2/labels.json

# Run the exported ONNX model locally on CPU.
models/noisy_drone_rf_v2/run_onnx_inference.sh --providers CPUExecutionProvider

# Inspect the strongest non-Noise class when the raw model top-1 is Noise.
models/noisy_drone_rf_v2/run_onnx_inference.sh \
  --providers CPUExecutionProvider \
  --decision-mode non-noise

# Scan a raw IQ capture, score burst windows by the target class, and classify the best window.
models/noisy_drone_rf_v2/run_onnx_inference.sh \
  --iq-file outputs/rx_debug.npy \
  --target-class FutabaT14 \
  --window-score-mode target \
  --decision-mode non-noise \
  --providers CPUExecutionProvider

# Run one high-SNR dataset sample per class and print a readable table.
models/noisy_drone_rf_v2/run_onnx_inference.sh \
  --class-sweep \
  --dataset-dir /data/rameyjm7/datasets/NoisyDroneRFv2 \
  --min-snr 20 \
  --samples-per-class 1 \
  --max-predictions 8 \
  --format table \
  --providers CPUExecutionProvider
```

Suggested end-to-end flow:

1. Install the package.
2. Download datasets or point `configs/local_data_paths.yaml` / workflow configs at local data.
3. Train or evaluate with `rfsi`.
4. Export artifacts and comparison tables.
5. Reproduce the headline metrics table.
6. Run the live SDR classifier.
7. Deploy the exported ONNX model through TensorRT / Jetson.

Older notebooks are retained, but the largest NoisyDroneRFv2 and comparison notebooks now act
as thin wrappers around reusable Python modules.

Legacy RML2016 entrypoint:

```bash
python -m rf_signal_intelligence --mode evaluate_only
```

Or installed console script:

```bash
rf-signal-intelligence --mode evaluate_only
```

Supported modes:
- `train`
- `train_continuously`
- `evaluate_only` (default)

Useful flags:
- `--data-path <path-to-RML2016.10a_dict.pkl>`
- `--model-name <artifact-prefix>`
- `--models-dir <output-dir>`
- `--stats-dir <output-dir>`
- `--outputs-dir <output-root>`

Defaults:
- Stats/logs are written under `outputs/`.
- If `models/<model-name>.keras` already exists, CLI will use it by default for compatibility.
- Otherwise, model artifacts default to `outputs/models/`.

## Testing

Fast checks:

```bash
ruff check src tests
pytest -q -m "not integration"
```

Full reproducibility command:

```bash
ruff check src tests && pytest -q -m "not integration" && pytest -q -m integration -rs
```

Integration checks (local datasets/models required):

```bash
pytest -q -m integration
```

CI integration artifact policy:
- Default policy is `skip_if_missing` (integration tests skip if artifacts are unavailable).
- To enforce artifacts and fail instead of skip, set:

```bash
export INTEGRATION_ARTIFACT_POLICY=require
```

Registry-driven smoke evaluation:

```bash
python scripts/smoke_eval_registry.py
python scripts/smoke_eval_registry.py --with-data --require-artifacts
```

To populate checksum values in registries from local artifacts:

```bash
python scripts/update_registry_checksums.py
```

## Docker

From `docker/`:

```bash
make build
make run
```

See [`docker/README.md`](docker/README.md) for Docker Hub and Apptainer/HPC usage.

## Notes

- Large datasets and model artifacts are expected; this repository is data-heavy.
- Working notebooks can produce uncommitted changes during experimentation.
- Release/tag process is documented in `RELEASE.md`.

## Citation

Ramey, J. M., and Goda, P. (2025). Wireless Signal Classification via Deep Learning.
Maintained by Jacob M. Ramey.
GitHub: https://github.com/rameyjm7/rf-signal-intelligence
