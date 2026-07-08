#!/usr/bin/env python3
"""Pipeline converted from the legacy 41_evaluation_rml2018 workflow."""

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
# Cell 1 : TEST - RML2018 evaluation (31-aligned data prep + pinned best checkpoint)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from rf_signal_intelligence.data.rml import load_rml2018_per_snr, load_rml2018_split
from rf_signal_intelligence.plots import per_class_accuracy_by_snr, write_per_class_snr_csv

notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    dcfg = cfg.get('datasets', {}).get('rml2018', {})
    h5_path = Path(dcfg.get('hdf5', '/scratch/rameyjm7/datasets/RML2018/GOLD_XYZ_OSC.0001_1024.hdf5'))
    classes_path = Path(dcfg.get('classes', '/scratch/rameyjm7/datasets/RML2018/classes.txt'))
else:
    h5_path = Path('/scratch/rameyjm7/datasets/RML2018/GOLD_XYZ_OSC.0001_1024.hdf5')
    classes_path = Path('/scratch/rameyjm7/datasets/RML2018/classes.txt')

best_ckpt_txt = project_root / 'models' / 'rml2018' / 'checkpoints' / 'best_checkpoint.txt'
default_model = project_root / 'models' / 'rml2018' / 'rml2018_lstm_rnn.keras'
if best_ckpt_txt.exists():
    candidate = Path(best_ckpt_txt.read_text().strip())
    model_path = candidate if candidate.exists() else default_model
else:
    model_path = default_model

outputs_dir = project_root / 'outputs' / 'rml2018_eval'
outputs_dir.mkdir(parents=True, exist_ok=True)
print('RML2018 dataset:', h5_path)
print('RML2018 classes:', classes_path)
print('RML2018 model  :', model_path)
print('Outputs dir   :', outputs_dir)
assert h5_path.exists(), f'Missing dataset: {h5_path}'
assert classes_path.exists(), f'Missing classes file: {classes_path}'
assert model_path.exists(), f'Missing model: {model_path}'

# %% Cell 2
# Cell 2 : Build 31-aligned eval split (SNR > -6 dB, <= 30 dB, stratified random split)
SNR_MIN_DB = -6
SNR_MAX_DB = 30
MAX_SAMPLES_PER_CLASS = 3000
TEST_SPLIT = 0.20
RANDOM_STATE = 42

X_all, y_all, class_list = load_rml2018_split(
    h5_path,
    classes_path,
    snr_min_db=SNR_MIN_DB,
    snr_max_db=SNR_MAX_DB,
    max_per_class=MAX_SAMPLES_PER_CLASS,
    random_state=RANDOM_STATE,
)
le = LabelEncoder()
y_all_enc = le.fit_transform(y_all)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all,
    y_all_enc,
    test_size=TEST_SPLIT,
    stratify=y_all_enc,
    random_state=RANDOM_STATE,
)
X_eval, y_eval = X_te, y_te
print('SNR filter: snr >', SNR_MIN_DB, 'and snr <=', SNR_MAX_DB)
print('Train shape:', X_tr.shape)
print('Eval shape :', X_eval.shape)
print('Num classes:', len(le.classes_))

# %% Cell 3
# Cell 3 : Evaluate model and plot confusion matrix
model = load_model(model_path, compile=False)
y_prob = model.predict(X_eval, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
print('Accuracy:', accuracy_score(y_eval, y_pred))
print(classification_report(y_eval, y_pred, target_names=le.classes_, zero_division=0))

cm = confusion_matrix(y_eval, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RML2018 Confusion Matrix')
plt.tight_layout()
cm_png = outputs_dir / '41_rml2018_confusion_matrix.png'
plt.savefig(cm_png, dpi=180)
print('Saved confusion matrix:', cm_png)
_save_current_figure("cell_03_figure_01.png")

# %% Cell 4
# Cell 4 : Plot and save accuracy per SNR per modulation
MAX_PER_CLASS_PER_SNR = 200
X_snr_eval, y_snr_labels, snr_vals, _ = load_rml2018_per_snr(
    h5_path,
    classes_path,
    snr_min_db=SNR_MIN_DB,
    snr_max_db=SNR_MAX_DB,
    max_per_class_per_snr=MAX_PER_CLASS_PER_SNR,
)
y_snr_true = le.transform(y_snr_labels)
y_snr_pred = np.argmax(model.predict(X_snr_eval, verbose=0), axis=1)
snr_unique, acc_grid = per_class_accuracy_by_snr(y_snr_true, y_snr_pred, snr_vals, n_classes=len(le.classes_))

plt.figure(figsize=(16, 10))
sns.heatmap(acc_grid / 100.0, cmap='viridis', vmin=0.0, vmax=1.0, xticklabels=snr_unique, yticklabels=le.classes_, cbar_kws={'label': 'Accuracy'})
plt.title('RML2018 Accuracy per SNR per Modulation')
plt.xlabel('SNR (dB)')
plt.ylabel('Modulation')
plt.tight_layout()
snr_mod_png = outputs_dir / '41_rml2018_accuracy_per_snr_per_modulation.png'
plt.savefig(snr_mod_png, dpi=180)
print('Saved per-SNR/per-modulation heatmap:', snr_mod_png)
_save_current_figure("cell_04_figure_02.png")
print('Saved per-SNR/per-modulation table:', write_per_class_snr_csv(outputs_dir / '41_rml2018_accuracy_per_snr_per_modulation.csv', list(le.classes_), snr_unique, acc_grid))
print('Per-SNR eval sample count:', X_snr_eval.shape[0])

# %% Cell 5
# Cell 5 : Plot and save line charts (overall accuracy vs SNR, per-modulation vs SNR)
overall = []
for snr in snr_unique:
    mask = snr_vals == snr
    overall.append(float(np.mean(y_snr_pred[mask] == y_snr_true[mask])) * 100.0)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].plot(snr_unique, overall, marker='o', color='blue')
axes[0].set_title('Recognition Accuracy vs. SNR (RML2018)')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].grid(True, alpha=0.4)
for idx, mod in enumerate(le.classes_):
    axes[1].plot(snr_unique, acc_grid[idx], marker='o', linewidth=1.0, label=mod)
axes[1].set_title('Accuracy vs. SNR per Modulation Type')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
plt.tight_layout()
png = outputs_dir / '41_rml2018_accuracy_vs_snr_line_plots.png'
plt.savefig(png, dpi=180)
print('Saved line charts:', png)
_save_current_figure("cell_05_figure_03.png")

# %% Cell 6
# Cell 6 : Load and plot continuation training curves from outputs/rml2018 history
import json

history_candidates = sorted((project_root / 'outputs' / 'rml2018').glob('*history*.json'))
if not history_candidates:
    print('No RML2018 history JSON files found under outputs/rml2018.')
else:
    history_path = history_candidates[-1]
    history = json.loads(history_path.read_text())
    print('Loaded history:', history_path)
    plt.figure(figsize=(10, 5))
    for key in ('accuracy', 'val_accuracy'):
        if key in history:
            plt.plot(history[key], label=key)
    plt.title('RML2018 Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    png = outputs_dir / '41_rml2018_training_curves.png'
    plt.savefig(png, dpi=180)
    print('Saved training curves:', png)
    _save_current_figure("cell_06_figure_04.png")
