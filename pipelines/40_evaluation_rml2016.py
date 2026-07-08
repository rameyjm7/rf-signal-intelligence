#!/usr/bin/env python3
"""Pipeline converted from the legacy 40_evaluation_rml2016 workflow."""

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from rf_signal_intelligence.data.rml import load_rml2016_pickle, rml2016_arrays
from rf_signal_intelligence.plots import accuracy_by_snr, per_class_accuracy_by_snr, write_overall_snr_csv, write_per_class_snr_csv

notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text())
    pkl_path = Path(cfg.get('datasets', {}).get('rml2016', {}).get('pkl', '/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl'))
else:
    pkl_path = Path('/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl')

model_path = project_root / 'models' / 'rml2016' / 'rml2016_lstm_rnn_2024.keras'
outputs_dir = project_root / 'outputs' / 'rml2016_eval'
outputs_dir.mkdir(parents=True, exist_ok=True)

print('RML2016 dataset:', pkl_path)
print('RML2016 model:', model_path)
print('Outputs dir:', outputs_dir)
assert pkl_path.exists(), f'Missing dataset: {pkl_path}'
assert model_path.exists(), f'Missing model: {model_path}'

# %% Cell 2
# Cell 2 : Load dataset and prepare train/test split with IQ+SNR features
data = load_rml2016_pickle(pkl_path)
X, y_text, class_names = rml2016_arrays(data)

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_test shape:', X_test.shape)
print('Classes:', list(encoder.classes_))

# %% Cell 3
# Cell 3 : Run model inference and print metrics
model = load_model(model_path, compile=False)
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

# %% Cell 4
# Cell 4 : Plot confusion matrix for RML2016
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RML2016 Confusion Matrix')
plt.tight_layout()
cm_png = outputs_dir / '40_rml2016_confusion_matrix.png'
plt.savefig(cm_png, dpi=180)
print('Saved confusion matrix:', cm_png)
_save_current_figure("cell_04_figure_01.png")

# %% Cell 5
# Cell 5 : Plot and save accuracy vs SNR (overall and per modulation)
snr_test = X_test[:, 0, 2].astype(int)
snr_values, overall_acc = accuracy_by_snr(y_test, y_pred, snr_test)
_, per_mod_acc = per_class_accuracy_by_snr(y_test, y_pred, snr_test, n_classes=len(encoder.classes_))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].plot(snr_values, overall_acc, marker='o', color='blue')
axes[0].set_title('Recognition Accuracy vs. SNR')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].grid(True, alpha=0.4)

for i, mod in enumerate(encoder.classes_):
    axes[1].plot(snr_values, per_mod_acc[i], marker='o', linewidth=1.4, label=mod)
axes[1].set_title('Accuracy vs. SNR per Modulation Type')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
plt.tight_layout()

snr_plots_png = outputs_dir / '40_rml2016_accuracy_vs_snr_plots.png'
plt.savefig(snr_plots_png, dpi=180)
print('Saved SNR plots:', snr_plots_png)
_save_current_figure("cell_05_figure_02.png")

print('Saved overall SNR accuracy table:', write_overall_snr_csv(outputs_dir / '40_rml2016_accuracy_vs_snr.csv', snr_values, overall_acc))
print('Saved per-modulation SNR accuracy table:', write_per_class_snr_csv(outputs_dir / '40_rml2016_accuracy_vs_snr_per_modulation.csv', list(encoder.classes_), snr_values, per_mod_acc))
