#!/usr/bin/env python3
"""Pipeline converted from the legacy 42_evaluation_deepradar2022 workflow."""

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
# Cell 1 : Import dependencies and resolve local config paths
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from rf_signal_intelligence.data.deepradar import load_deepradar_test_split
from rf_signal_intelligence.plots import accuracy_by_snr, per_class_accuracy_by_snr, write_overall_snr_csv, write_per_class_snr_csv

notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text())
    dcfg = cfg.get('datasets', {}).get('deepradar2022', {})
    x_path = Path(dcfg.get('x_test', '/scratch/rameyjm7/datasets/DeepRadar2022/X_test.mat'))
    y_path = Path(dcfg.get('y_test', '/scratch/rameyjm7/datasets/DeepRadar2022/Y_test.mat'))
    lbl_path = Path(dcfg.get('lbl_test', '/scratch/rameyjm7/datasets/DeepRadar2022/lbl_test.mat'))
else:
    x_path = Path('/scratch/rameyjm7/datasets/DeepRadar2022/X_test.mat')
    y_path = Path('/scratch/rameyjm7/datasets/DeepRadar2022/Y_test.mat')
    lbl_path = Path('/scratch/rameyjm7/datasets/DeepRadar2022/lbl_test.mat')

model_path = project_root / 'models' / 'deepradar2022' / 'deepradar2022_cnn_bilstm_final.keras'
outputs_dir = project_root / 'outputs' / 'deepradar2022_eval'
outputs_dir.mkdir(parents=True, exist_ok=True)
print('DeepRadar2022 X:', x_path)
print('DeepRadar2022 Y:', y_path)
print('DeepRadar2022 lbl:', lbl_path)
print('DeepRadar2022 model:', model_path)
print('Outputs dir:', outputs_dir)
assert x_path.exists() and y_path.exists() and lbl_path.exists(), 'Missing DeepRadar2022 files.'
assert model_path.exists(), f'Missing model: {model_path}'

# %% Cell 2
# Cell 2 : Load test split and build model input features
X_eval, y_true, snr_test = load_deepradar_test_split(x_path, y_path, lbl_path)
print('X_eval shape:', X_eval.shape)
print('class count:', int(y_true.max()) + 1)
print('snr_test available:', snr_test is not None)

# %% Cell 3
# Cell 3 : Run model inference and print metrics
model = load_model(model_path, compile=False)
y_prob = model.predict(X_eval, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
print('Accuracy:', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, zero_division=0))

# %% Cell 4
# Cell 4 : Plot confusion matrix for DeepRadar2022
n_classes = int(max(y_true.max(), y_pred.max()) + 1)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeepRadar2022 Confusion Matrix')
plt.tight_layout()
cm_png = outputs_dir / '42_deepradar2022_confusion_matrix.png'
plt.savefig(cm_png, dpi=180)
print('Saved confusion matrix:', cm_png)
_save_current_figure("cell_04_figure_01.png")

# %% Cell 5
# Cell 5 : Plot and save SNR line charts (overall and per class)
if snr_test is None:
    print('SNR metadata not available; skipping SNR line charts.')
else:
    snr_unique, overall_acc = accuracy_by_snr(y_true, y_pred, snr_test)
    _, per_class = per_class_accuracy_by_snr(y_true, y_pred, snr_test, n_classes=int(max(y_true.max(), y_pred.max()) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].plot(snr_unique, overall_acc, marker='o', color='blue')
    axes[0].set_title('Recognition Accuracy vs. SNR (DeepRadar2022)')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, alpha=0.4)

    for c in range(per_class.shape[0]):
        axes[1].plot(snr_unique, per_class[c], marker='o', linewidth=1.0, label=f'class_{c}')
    axes[1].set_title('Accuracy vs. SNR per Class (DeepRadar2022)')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    plt.tight_layout()
    png = outputs_dir / '42_deepradar2022_accuracy_vs_snr_line_plots.png'
    plt.savefig(png, dpi=180)
    print('Saved line charts:', png)
    _save_current_figure("cell_05_figure_02.png")

    print('Saved overall line data:', write_overall_snr_csv(outputs_dir / '42_deepradar2022_accuracy_vs_snr_line.csv', snr_unique, overall_acc))
    print('Saved per-class line data:', write_per_class_snr_csv(outputs_dir / '42_deepradar2022_accuracy_vs_snr_per_class.csv', [f'class_{i}' for i in range(per_class.shape[0])], snr_unique, per_class))
