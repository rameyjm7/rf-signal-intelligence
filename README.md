# ML Wireless Signal Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![Docker](https://img.shields.io/badge/Docker-GPU--Ready-blue)

**Maintainer:** Jacob M. Ramey  
LinkedIn: https://www.linkedin.com/in/rameyjm/

Deep-learning workflows for wireless modulation classification, centered on RNN-LSTM models and related experiments.

## Table of Contents
- [Overview](#overview)
- [Current Progress (April 2026)](#current-progress-april-2026)
- [Repository Layout](#repository-layout)
- [Datasets](#datasets)
- [Results: RML2016](#results-rml2016)
- [Results: RML2018](#results-rml2018)
- [Results: DeepRadar2022](#results-deepradar2022)
- [Requirements](#requirements)
- [Local Setup](#local-setup)
- [CLI Usage](#cli-usage)
- [Docker](#docker)
- [Notes](#notes)
- [Citation](#citation)

## Overview

This repository contains training pipelines, evaluation notebooks, saved models, and documentation for deep-learning-based wireless signal classification.

Implemented workflows include:
- LSTM and BiLSTM architectures for raw I/Q modeling
- CNN + recurrent hybrids for time-frequency structure
- Cross-dataset experimentation on RML2016, RML2018, and DeepRadar2022
- GPU-ready execution via Docker/Apptainer

## Current Progress (April 2026)

- RML2018 was rebaselined with a new continuation training run; best checkpoint evaluation reached `0.8295` accuracy on the current filtered protocol used in notebook evaluation.
- Cross-dataset ensemble evaluation (`43`) is now aligned with pinned best-checkpoint loading and class-order calibration, producing stable combined results (recent run: `0.96` overall on the sampled combined set).
- Notebook naming and pipeline flow were standardized:
  - training: `30_lstm_rml2016.ipynb`, `31_lstm_rml2018.ipynb`, `32_lstm_deepradar2022.ipynb`
  - evaluation: `40_evaluation_rml2016.ipynb`, `41_evaluation_rml2018.ipynb`, `42_evaluation_deepradar2022.ipynb`, `43_evaluation_cross_dataset_ensemble.ipynb`, `50_evaluation_comparison.ipynb`
- Evaluation notebooks now save local artifacts under `outputs/` (confusion matrices, classification reports, SNR charts, and training-curve plots) so notebook outputs can be cleared while retaining reproducible figures/tables.
- Artifact review notebook added: `notebooks/99_outputs_artifact_review.ipynb` for consolidating saved outputs before README/report updates.

### Latest Notebook 50 Run (Local)

| Dataset | Eval protocol | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---|---:|---:|---:|---:|
| DeepRadar2022 | All SNR levels | 0.8433 | 0.84 | 0.84 | 156,400 |
| RML2016 | All SNR levels | 0.67 | 0.69 | 0.69 | 44,000 |
| RML2016 | SNR > 5 dB | 0.93 | 0.92 | 0.92 | 15,332 |
| RML2018 | Highest-SNR class-balanced slice (mapping-calibrated) | 0.9465 | 0.94 | 0.94 | 4,800 |

## Repository Layout

```text
src/ml_wireless_classification/   Python package
  core/                           Maintained runtime/training components
  models/                         Maintained model definitions
  legacy/                         Experimental utilities kept for archive notebooks
  base/                           Backward-compatible import wrappers
configs/                          Dataset and model registries (YAML)
data/                             Datasets (RML2016, RML2018, DeepRadar2022)
models/                           Saved model artifacts
outputs/                          Generated runtime outputs (stats, logs, new artifacts)
notebooks/                        Reproducible notebooks
docker/                           Docker and Apptainer build/runtime files
docs/                             Project reports and papers
tests/                            Test and integration checks
archive/                          Archived experiments and prototype artifacts
```

## Datasets

### RML2016.10A
11 modulation types, SNR from -20 dB to +18 dB.

### RML2018.01A
24 classes; used for larger-scale training/evaluation and cross-dataset checks.

### DeepRadar2022
Radar waveform dataset used for CNN-BiLSTM style modeling and transfer evaluation.

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

Run as module:

```bash
python -m ml_wireless_classification --mode evaluate_only
```

Or installed console script:

```bash
ml-wireless-classification --mode evaluate_only
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
GitHub: https://github.com/rameyjm7/ML-wireless-signal-classification
