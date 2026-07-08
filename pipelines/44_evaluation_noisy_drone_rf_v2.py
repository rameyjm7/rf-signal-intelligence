#!/usr/bin/env python3
"""Pipeline converted from the legacy 44_evaluation_noisy_drone_rf_v2 workflow."""

from __future__ import annotations

from pathlib import Path


# Pipeline artifact helpers: converted from notebook workflows, so plot displays are saved.
def _pipeline_artifact_dir() -> Path:
    path = Path("outputs/pipeline_figures") / Path(__file__).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_current_figure(filename: str) -> None:
    import matplotlib.pyplot as plt

    path = _pipeline_artifact_dir() / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved figure: {path}")

# %% Cell 1
# Cell 1 : Import dependencies and resolve VGG full-complex evaluation paths
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from rf_signal_intelligence.data.noisy_drone import build_manifest, label_names_from_class_stats
from rf_signal_intelligence.features.spectrogram import SpectrogramConfig
from rf_signal_intelligence.workflows.noisy_drone_vgg import prepare_spectrogram
from rf_signal_intelligence.plots import accuracy_by_snr, per_class_accuracy_by_snr, write_overall_snr_csv, write_per_class_snr_csv

notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    dcfg = cfg.get('datasets', {}).get('noisy_drone_rf_v2', {}) or {}
    data_dir = Path(dcfg.get('data_dir', '/scratch/rameyjm7/datasets/NoisyDroneRFv2'))
else:
    data_dir = Path('/scratch/rameyjm7/datasets/NoisyDroneRFv2')

model_dir = project_root / 'models' / 'noisy_drone_rf_v2'
vgg_best_path = model_dir / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras'
vgg_final_path = model_dir / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_final.keras'
model_path = vgg_final_path if vgg_final_path.exists() else vgg_best_path
outputs_dir = project_root / 'outputs' / 'noisy_drone_rf_v2_eval'
outputs_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 1961
MAX_IQ_SAMPLES = 1048576
MIN_SNR_DB = -6
DATA_FRACTION = 0.25
BATCH_SIZE = 8
SPEC_NFFT = 1024
SPEC_HOP = 1024
SPEC_TIME_BINS = 1024
BALANCED_EVAL = True
cache_dir = Path('/scratch/rameyjm7/ML-wireless-signal-classification/cache/noisy_drone_rf_v2/spectrogram_full_complex_cache')
cache_dir.mkdir(parents=True, exist_ok=True)

print('Noisy Drone RF v2 data:', data_dir)
print('Noisy Drone RF v2 VGG model:', model_path)
print('Outputs dir:', outputs_dir)
print('Spectrogram cache:', cache_dir)
assert data_dir.exists(), f'Missing NoisyDroneRFv2 directory: {data_dir}'
assert model_path.exists(), f'Missing VGG model: {model_path}'

# %% Cell 2
# Cell 2 : Build the Noisy Drone RF v2 eval manifest and full-complex spectrogram helpers
records = build_manifest(data_dir, min_snr_db=MIN_SNR_DB, data_fraction=DATA_FRACTION, random_state=RANDOM_STATE)
assert records, f'No matching .pt files found under {data_dir}'
classes = sorted({record.target_raw for record in records})
label_names = label_names_from_class_stats(data_dir, classes)
num_classes = len(classes)
spec_config = SpectrogramConfig(sample_len=MAX_IQ_SAMPLES, nfft=SPEC_NFFT, hop=SPEC_HOP, time_bins=SPEC_TIME_BINS)

labels = np.array([record.label_idx for record in records], dtype=np.int64)
idx = np.arange(len(records))
train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=labels)
test_records = [records[int(i)] for i in test_idx]

if BALANCED_EVAL:
    rng = np.random.default_rng(RANDOM_STATE)
    by_label = {label: [record for record in test_records if record.label_idx == label] for label in sorted(set(labels))}
    target_n = min(len(group) for group in by_label.values())
    balanced_test_records = []
    for label, group in by_label.items():
        chosen = rng.choice(len(group), size=target_n, replace=False)
        balanced_test_records.extend(group[int(i)] for i in chosen)
else:
    balanced_test_records = None

print('Samples selected:', len(records))
print('Class names:', label_names)
print('Test natural counts:', {int(k): int(v) for k, v in zip(*np.unique([r.label_idx for r in test_records], return_counts=True))})
if balanced_test_records is not None:
    print('Test balanced counts:', {int(k): int(v) for k, v in zip(*np.unique([r.label_idx for r in balanced_test_records], return_counts=True))})
print('VGG full-complex input shape:', spec_config.input_shape)

# %% Cell 3
# Cell 3 : Evaluate the canonical VGG full-complex spectrogram model
vgg_eval_model = load_model(model_path, compile=False)
print('Loaded model:', model_path)
print('Model input shape:', vgg_eval_model.input_shape)

def predict_records(sample_records, name='eval'):
    probs = []
    for i, record in enumerate(sample_records, start=1):
        x = prepare_spectrogram(record.filepath, snr=record.snr, cache_dir=cache_dir, config=spec_config)
        probs.append(vgg_eval_model.predict(x[None, ...], batch_size=BATCH_SIZE, verbose=0)[0])
        if i <= 3 or i % 100 == 0 or i == len(sample_records):
            print(f'{name}: {i}/{len(sample_records)}', flush=True)
    return np.asarray(probs, dtype=np.float32)

natural_probs = predict_records(test_records, name='natural')
y_true = np.array([record.label_idx for record in test_records], dtype=np.int64)
y_pred = natural_probs.argmax(axis=1)

natural_metrics = {
    'accuracy': float(accuracy_score(y_true, y_pred)),
    'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
}
print('Noisy Drone RF v2 VGG natural report')
print(natural_metrics)
print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
pd.DataFrame(classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)).transpose().to_csv(outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_classification_report.csv')

balanced_metrics = None
if balanced_test_records is not None:
    balanced_probs = predict_records(balanced_test_records, name='balanced')
    y_balanced_true = np.array([record.label_idx for record in balanced_test_records], dtype=np.int64)
    y_balanced_pred = balanced_probs.argmax(axis=1)
    balanced_metrics = {
        'balanced_accuracy': float(accuracy_score(y_balanced_true, y_balanced_pred)),
        'balanced_macro_f1': float(f1_score(y_balanced_true, y_balanced_pred, average='macro', zero_division=0)),
        'balanced_weighted_f1': float(f1_score(y_balanced_true, y_balanced_pred, average='weighted', zero_division=0)),
    }
    print('Noisy Drone RF v2 VGG balanced report')
    print(balanced_metrics)
    print(classification_report(y_balanced_true, y_balanced_pred, target_names=label_names, zero_division=0))

metrics = {'model': 'noisy_drone_rf_v2_vgg_full_complex_spectrogram', 'model_path': str(model_path), 'test_samples': len(test_records), 'label_names': label_names, **natural_metrics}
if balanced_metrics is not None:
    metrics.update(balanced_metrics)
metrics_path = outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_metrics.json'
metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('Saved metrics:', metrics_path)

# %% Cell 4
# Cell 4 : Plot natural and balanced confusion matrices for the VGG full-complex model
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Noisy Drone RF v2 VGG Confusion Matrix')
plt.tight_layout()
cm_png = outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_confusion_matrix.png'
plt.savefig(cm_png, dpi=180)
print('Saved confusion matrix:', cm_png)
_save_current_figure("cell_04_figure_01.png")

if balanced_test_records is not None:
    balanced_cm = confusion_matrix(y_balanced_true, y_balanced_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(10, 8))
    sns.heatmap(balanced_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Noisy Drone RF v2 VGG Balanced Confusion Matrix')
    plt.tight_layout()
    balanced_cm_png = outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_balanced_confusion_matrix.png'
    plt.savefig(balanced_cm_png, dpi=180)
    print('Saved balanced confusion matrix:', balanced_cm_png)
    _save_current_figure("cell_04_figure_02.png")

# %% Cell 5
# Cell 5 : Plot and save VGG full-complex accuracy across SNR, overall and per class
snr_test = np.array([record.snr for record in test_records], dtype=np.int64)
snr_values, overall_acc = accuracy_by_snr(y_true, y_pred, snr_test)
_, per_class = per_class_accuracy_by_snr(y_true, y_pred, snr_test, n_classes=num_classes)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].plot(snr_values, overall_acc, marker='o', color='blue')
axes[0].set_title('Noisy Drone RF v2 VGG Accuracy vs. SNR')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].grid(True, alpha=0.4)
for idx, name in enumerate(label_names):
    axes[1].plot(snr_values, per_class[idx], marker='o', linewidth=1.2, label=name)
axes[1].set_title('Noisy Drone RF v2 VGG Accuracy vs. SNR per Class')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
plt.tight_layout()
png = outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr.png'
plt.savefig(png, dpi=180)
print('Saved SNR chart:', png)
_save_current_figure("cell_05_figure_03.png")

print('Saved SNR table:', write_overall_snr_csv(outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr.csv', snr_values, overall_acc))
print('Saved per-class SNR table:', write_per_class_snr_csv(outputs_dir / '44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr_per_class.csv', label_names, snr_values, per_class))
