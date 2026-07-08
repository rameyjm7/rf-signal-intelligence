#!/usr/bin/env python3
"""Pipeline converted from the legacy 50_evaluation_comparison workflow."""

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
# Cell 1 : TEST - DeepRadar2022 model evaluation
# https://www.kaggle.com/jacobramey
# https://github.com/rameyjm7

import os
import numpy as np
import h5py, scipy.io as sio
import importlib
import matplotlib as mpl
if not hasattr(mpl, 'backends'):
    mpl.backends = importlib.import_module('matplotlib.backends')
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from pathlib import Path
import yaml

# ------------------------------
# Resolve notebook directory safely
# ------------------------------
notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

print(f"Pipeline directory: {notebook_dir}")
print(f"Project root: {project_root}")
outputs_dir_50 = project_root / "outputs" / "50_evaluation_comparison"
outputs_dir_50.mkdir(parents=True, exist_ok=True)
print(f"Outputs dir: {outputs_dir_50}")

# ------------------------------
# Load DeepRadar2022 dataset
# ------------------------------
cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text())
    path = Path(local_cfg.get('dataset_root', '/scratch/rameyjm7/datasets')) / 'DeepRadar2022'
else:
    path = Path('/scratch/rameyjm7/datasets/DeepRadar2022')
print(f"Loading DeepRadar2022 from: {path}")

def load_h5(filepath, key):
    with h5py.File(filepath, "r") as f:
        return np.array(f[key], dtype="float32").T

def load_mat(filepath, key):
    d = sio.loadmat(filepath)
    return d[key]

X_test  = load_h5(path / "X_test.mat", "X_test")
Y_test  = load_mat(path / "Y_test.mat", "Y_test")
lbl_test = load_mat(path / "lbl_test.mat", "lbl_test")

# Extract modulation class and SNR
cls_test, snr_test = lbl_test[:,0].astype(int)-1, lbl_test[:,1]

# ------------------------------
# Normalize IQ per sample
# ------------------------------
def normalize_iq(X):
    Xn = np.empty_like(X)
    for i in range(X.shape[0]):
        scale = np.max(np.abs(X[i])) + 1e-12
        Xn[i] = X[i] / scale
    return Xn

X_test = normalize_iq(X_test)

# ------------------------------
# Append SNR as a third channel
# ------------------------------
def append_snr_feature(X, snr):
    X_out = []
    for i in range(X.shape[0]):
        snr_col = np.full((X.shape[1], 1), snr[i] / 20.0)
        X_out.append(np.concatenate([X[i], snr_col], axis=1))
    return np.array(X_out, dtype=np.float32)

X_test = append_snr_feature(X_test, snr_test)

# ------------------------------
# Load pre-trained model
# ------------------------------
model_path = project_root / "models" / "deepradar2022" / "deepradar2022_cnn_bilstm_final.keras"
print(f"Loading model from: {model_path}")
model = load_model(model_path)

# ------------------------------
# Evaluate model
# ------------------------------
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\n✅ Model Test Accuracy (all SNRs): {acc*100:.2f}%")

# ------------------------------
# Predictions and metrics
# ------------------------------
Y_pred = model.predict(X_test, verbose=0)
y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(Y_pred, axis=1)

# ------------------------------
# Confusion Matrix
# ------------------------------
labels = [
    "LFM", "2FSK", "4FSK", "8FSK", "Costas", "2PSK", "4PSK", "8PSK",
    "Barker", "Huffman", "Frank", "P1", "P2", "P3", "P4", "Px",
    "Zadoff-Chu", "T1", "T2", "T3", "T4", "NM", "Noise"
]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues", cbar=True,
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (CNN + BiLSTM, DeepRadar2022)")
plt.tight_layout()
_save_current_figure("cell_01_figure_01.png")

print("\nClassification Report (All SNRs):")
print(classification_report(y_true, y_pred, target_names=labels))

# ------------------------------
# Accuracy vs SNR
# ------------------------------
unique_snrs = sorted(np.unique(snr_test))
acc_snr = []
for snr in unique_snrs:
    idx = np.where(snr_test == snr)[0]
    acc_snr.append(accuracy_score(y_true[idx], y_pred[idx]) * 100)

plt.figure(figsize=(8,5))
plt.plot(unique_snrs, acc_snr, 'b-o')
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Recognition Accuracy vs SNR (DeepRadar2022 CNN+BiLSTM)")
plt.grid(True)
plt.tight_layout()
_save_current_figure("cell_01_figure_02.png")


# ------------------------------
# Accuracy vs SNR per class (DeepRadar2022)
# ------------------------------
class_ids = np.array(sorted(np.unique(y_true)), dtype=int)
per_class_acc = np.full((len(class_ids), len(unique_snrs)), np.nan, dtype=np.float32)
for i, cls in enumerate(class_ids):
    cls_mask = y_true == cls
    for j, snr in enumerate(unique_snrs):
        idx = np.where((snr_test == snr) & cls_mask)[0]
        if len(idx) > 0:
            per_class_acc[i, j] = accuracy_score(y_true[idx], y_pred[idx]) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].plot(unique_snrs, acc_snr, marker='o', color='blue')
axes[0].set_title('Recognition Accuracy vs. SNR (DeepRadar2022)')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].grid(True, alpha=0.4)

for i, cls in enumerate(class_ids):
    axes[1].plot(unique_snrs, per_class_acc[i], marker='o', linewidth=1.0, label=f'class_{cls}')
axes[1].set_title('Accuracy vs. SNR per Class (DeepRadar2022)')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
plt.tight_layout()

png = outputs_dir_50 / '50_deepradar2022_accuracy_vs_snr_line_plots.png'
plt.savefig(png, dpi=180)
print('Saved line charts:', png)
_save_current_figure("cell_01_figure_03.png")

# %% Cell 2
# Cell 2 : TEST - RML2016 model evaluation
# https://www.kaggle.com/jacobramey
# https://github.com/rameyjm7

import pickle
import numpy as np
import importlib
import matplotlib as mpl
if not hasattr(mpl, 'backends'):
    mpl.backends = importlib.import_module('matplotlib.backends')
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from pathlib import Path
import yaml

# --------------------------------------------------------------
# Resolve model path
# --------------------------------------------------------------
project_root = str(Path.cwd().resolve().parent if Path.cwd().resolve().name == "pipelines" else Path.cwd().resolve())
model_path = os.path.join(project_root, "models", "rml2016", "rml2016_lstm_rnn_2024.keras")

print("Resolved model path:", model_path)
assert os.path.exists(model_path), f"Model file not found: {model_path}"

model = load_model(model_path)
print("Model loaded successfully.")
outputs_dir_50 = Path(project_root) / "outputs" / "50_evaluation_comparison"
outputs_dir_50.mkdir(parents=True, exist_ok=True)
print("Outputs dir:", outputs_dir_50)

# --------------------------------------------------------------
# Load RML2016.10a Dataset
# --------------------------------------------------------------
cfg_path = Path(project_root) / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text())
    pkl_path = local_cfg.get('datasets', {}).get('rml2016', {}).get('pkl', '/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl')
else:
    pkl_path = '/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl'
print("Loading dataset:", pkl_path)

with open(pkl_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# --------------------------------------------------------------
# Prepare Data (your exact provided format)
# --------------------------------------------------------------
def prepare_data(data):
    X, y, snrs = [], [], []

    for (mod_type, snr), signals in data.items():
        for signal in signals:
            # IQ: shape (128, 2)
            iq = np.vstack([signal[0], signal[1]]).T

            # SNR feature channel (raw SNR, consistent with your training format)
            snr_col = np.full((128, 1), snr, dtype=np.float32)

            combined = np.hstack([iq, snr_col])  # (128, 3)

            X.append(combined)
            y.append(mod_type)
            snrs.append(snr)   # keep real SNR for analysis

    X = np.array(X)
    y = np.array(y)
    snrs = np.array(snrs)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, y_encoded, snrs, test_size=0.2, random_state=42
    )

    # LSTM requires this shape already (128, 3)
    return X_train, X_test, y_train, y_test, snr_train, snr_test, encoder

# Prepare
X_train, X_test, y_train, y_test, snr_train, snr_test, encoder = prepare_data(data)

# --------------------------------------------------------------
# Evaluate model on full test set
# --------------------------------------------------------------
y_pred = np.argmax(model.predict(X_test, verbose=False), axis=1)

# Confusion matrix (ALL SNR)
cm_all = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (All SNR Levels)")
_save_current_figure("cell_02_figure_04.png")

print("\nClassification Report (All SNR Levels):")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# --------------------------------------------------------------
# Evaluate only SNR > 5 dB subset
# --------------------------------------------------------------
idx_high = np.where(snr_test > 5)[0]

X_high = X_test[idx_high]
y_high = y_test[idx_high]

print(f"\nSamples with SNR > 5 dB: {len(idx_high)}")

y_pred_high = np.argmax(model.predict(X_high, verbose=False), axis=1)

cm_high = confusion_matrix(y_high, y_pred_high)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_high, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (SNR > 5 dB)")
_save_current_figure("cell_02_figure_05.png")

print("\nClassification Report (SNR > 5 dB):")
print(classification_report(y_high, y_pred_high, target_names=encoder.classes_))



# --------------------------------------------------------------
# Accuracy vs SNR charts (RML2016)
# --------------------------------------------------------------
unique_snrs = np.array(sorted(np.unique(snr_test)), dtype=int)
overall_acc = []
per_mod_acc = np.full((len(encoder.classes_), len(unique_snrs)), np.nan, dtype=np.float32)

for j, snr in enumerate(unique_snrs):
    idx = np.where(snr_test == snr)[0]
    overall_acc.append(float(np.mean(y_pred[idx] == y_test[idx])) * 100.0)

for c in range(len(encoder.classes_)):
    cls_mask = y_test == c
    for j, snr in enumerate(unique_snrs):
        idx = np.where((snr_test == snr) & cls_mask)[0]
        if len(idx) > 0:
            per_mod_acc[c, j] = float(np.mean(y_pred[idx] == y_test[idx])) * 100.0

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].plot(unique_snrs, overall_acc, marker='o', color='blue')
axes[0].set_title('Recognition Accuracy vs. SNR (RML2016)')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].grid(True, alpha=0.4)

for i, mod in enumerate(encoder.classes_):
    axes[1].plot(unique_snrs, per_mod_acc[i], marker='o', linewidth=1.1, label=mod)
axes[1].set_title('Accuracy vs. SNR per Modulation Type (RML2016)')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
plt.tight_layout()

png = outputs_dir_50 / '50_rml2016_accuracy_vs_snr_line_plots.png'
plt.savefig(png, dpi=180)
print('Saved line charts:', png)
_save_current_figure("cell_02_figure_06.png")

# %% Cell 3
# Cell 3 : TEST - RML2018 model evaluation
# https://www.kaggle.com/jacobramey
# https://github.com/rameyjm7

from pathlib import Path
import ast
import re

import h5py
import numpy as np
import importlib
import matplotlib as mpl
if not hasattr(mpl, 'backends'):
    mpl.backends = importlib.import_module('matplotlib.backends')
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import json
from tensorflow.keras.models import load_model

# --------------------------------------------------------------
# Resolve paths and config
# --------------------------------------------------------------
notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text())
    dcfg = cfg.get('datasets', {}).get('rml2018', {})
    h5_path = Path(dcfg.get('hdf5', '/scratch/rameyjm7/datasets/RML2018/GOLD_XYZ_OSC.0001_1024.hdf5'))
    classes_path = Path(dcfg.get('classes', '/scratch/rameyjm7/datasets/RML2018/classes.txt'))
    classes_fixed_path = Path(dcfg.get('classes_fixed', '/scratch/rameyjm7/datasets/RML2018/classes-fixed.txt'))
else:
    h5_path = Path('/scratch/rameyjm7/datasets/RML2018/GOLD_XYZ_OSC.0001_1024.hdf5')
    classes_path = Path('/scratch/rameyjm7/datasets/RML2018/classes.txt')
    classes_fixed_path = Path('/scratch/rameyjm7/datasets/RML2018/classes-fixed.txt')

best_ckpt_txt = project_root / 'models' / 'rml2018' / 'checkpoints' / 'best_checkpoint.txt'
default_model_path = project_root / 'models' / 'rml2018' / 'rml2018_lstm_rnn.keras'
if best_ckpt_txt.exists():
    cand = Path(best_ckpt_txt.read_text().strip())
    model_path = cand if cand.exists() else default_model_path
else:
    model_path = default_model_path

print('RML2018 dataset:', h5_path)
print('RML2018 model:', model_path)
outputs_dir_50 = project_root / 'outputs' / '50_evaluation_comparison'
outputs_dir_50.mkdir(parents=True, exist_ok=True)
print('Outputs dir:', outputs_dir_50)

assert h5_path.exists(), f'Missing dataset: {h5_path}'
assert classes_path.exists(), f'Missing classes file: {classes_path}'
assert classes_fixed_path.exists(), f'Missing classes-fixed file: {classes_fixed_path}'
assert model_path.exists(), f'Missing model: {model_path}'

# --------------------------------------------------------------
# Build highest-SNR class-balanced eval split
# --------------------------------------------------------------
def parse_classes(path: Path):
    text = path.read_text()
    match = re.search(r'classes\s*=\s*(\[[\s\S]*?\])', text)
    if not match:
        raise ValueError(f'Could not parse classes from {path}')
    return ast.literal_eval(match.group(1))

classes_orig = parse_classes(classes_path)
classes_fixed = parse_classes(classes_fixed_path)
remap_fixed = np.array([classes_fixed.index(c) for c in classes_orig], dtype=np.int64)

with h5py.File(h5_path, 'r') as h5:
    X = h5['X']
    Y = h5['Y']
    Z = h5['Z']

    snr = Z[:, 0]
    max_snr = int(np.max(snr))
    max_idx = np.where(snr == max_snr)[0]
    y_max = np.argmax(Y[max_idx], axis=1)

    rng = np.random.default_rng(42)
    picked = []
    for cls in np.unique(y_max):
        cls_idx = max_idx[y_max == cls]
        k = min(200, len(cls_idx))
        picked.extend(rng.choice(cls_idx, size=k, replace=False).tolist())

    picked = np.array(sorted(picked), dtype=np.int64)
    x_iq = np.asarray(X[picked], dtype=np.float32)
    y_orig = np.argmax(np.asarray(Y[picked]), axis=1).astype(np.int64)
    snr_vals = np.asarray(Z[picked, 0], dtype=np.float32)

# Build alternate target encodings
y_fixed = remap_fixed[y_orig]
y_names = np.array([classes_orig[i] for i in y_orig])
le = LabelEncoder()
y_le = le.fit_transform(y_names)
classes_le = list(le.classes_)

snr_ch = np.repeat(snr_vals[:, None, None], x_iq.shape[1], axis=1)
X_eval = np.concatenate([x_iq, snr_ch], axis=2).astype(np.float32)

print('X_eval shape:', X_eval.shape)
print('max_snr used:', max_snr)

# --------------------------------------------------------------
# Evaluate model with automatic label-order calibration
# --------------------------------------------------------------
model = load_model(model_path, compile=False)
y_pred = np.argmax(model.predict(X_eval, verbose=0), axis=1)

acc_orig = accuracy_score(y_orig, y_pred)
acc_fixed = accuracy_score(y_fixed, y_pred)
acc_le = accuracy_score(y_le, y_pred)

candidates = [
    ('classes', y_orig, classes_orig, acc_orig),
    ('classes-fixed', y_fixed, classes_fixed, acc_fixed),
    ('labelencoder', y_le, classes_le, acc_le),
]
order_name, y_true, target_names, acc = max(candidates, key=lambda t: t[3])

print(f'RML2018 mapping calibration: acc_orig={acc_orig:.4f}, acc_fixed={acc_fixed:.4f}, acc_le={acc_le:.4f}')
print('Using mapping:', order_name)
print(f'RML2018 evaluation accuracy: {acc:.4f}')
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(target_names)))

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap='Blues')
plt.title(f'Confusion Matrix (RML2018, mapping={order_name})')
plt.xlabel('Predicted class index')
plt.ylabel('True class index')
plt.tight_layout()
cm_png = outputs_dir_50 / '50_rml2018_confusion_matrix.png'
plt.savefig(cm_png, dpi=180)
print('Saved confusion matrix:', cm_png)
_save_current_figure("cell_03_figure_07.png")

# Optional: load and plot continuation training curves from outputs/rml2018
history_dir = project_root / 'outputs' / 'rml2018'
history_files = sorted(history_dir.glob('*.history.json'))
if history_files:
    latest_hist = max(history_files, key=lambda p: p.stat().st_mtime)
    print('Using history file:', latest_hist)
    hist = json.loads(latest_hist.read_text())
    acc_h = hist.get('accuracy', [])
    val_acc_h = hist.get('val_accuracy', [])
    loss_h = hist.get('loss', [])
    val_loss_h = hist.get('val_loss', [])

    if acc_h and val_acc_h and loss_h and val_loss_h:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc_h, label='train_accuracy')
        plt.plot(val_acc_h, label='val_accuracy')
        plt.title(f'Continuation Accuracy ({len(acc_h)} epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss_h, label='train_loss')
        plt.plot(val_loss_h, label='val_loss')
        plt.title(f'Continuation Loss ({len(loss_h)} epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        curves_png = outputs_dir_50 / '50_rml2018_continuation_training_curves.png'
        plt.savefig(curves_png, dpi=180)
        print('Saved training curves:', curves_png)
        _save_current_figure("cell_03_figure_08.png")



# --------------------------------------------------------------
# Build broader per-SNR slice and plot line charts (RML2018)
# --------------------------------------------------------------
MAX_PER_CLASS_PER_SNR = 120

X_rows, y_rows, snr_rows = [], [], []
with h5py.File(h5_path, 'r') as h5:
    X_all = h5['X']
    Y_all = h5['Y']
    Z_all = h5['Z']

    buckets = {}
    for i in range(len(X_all)):
        s = int(Z_all[i, 0])
        cls = int(np.argmax(Y_all[i]))
        key = (cls, s)
        buckets.setdefault(key, 0)
        if buckets[key] >= MAX_PER_CLASS_PER_SNR:
            continue
        buckets[key] += 1

        iq = np.asarray(X_all[i], dtype=np.float32)
        snr_col = np.full((iq.shape[0], 1), s, dtype=np.float32)
        X_rows.append(np.concatenate([iq, snr_col], axis=1))
        y_rows.append(cls)
        snr_rows.append(s)

X_snr = np.asarray(X_rows, dtype=np.float32)
y_snr_orig = np.asarray(y_rows, dtype=np.int64)
snr_vals = np.asarray(snr_rows, dtype=np.int64)
if order_name == 'classes-fixed':
    y_snr = remap_fixed[y_snr_orig]
    snr_target_names = classes_fixed
elif order_name == 'labelencoder':
    y_snr_names = np.array([classes_orig[i] for i in y_snr_orig])
    y_snr = le.transform(y_snr_names)
    snr_target_names = classes_le
else:
    y_snr = y_snr_orig
    snr_target_names = classes_orig

y_snr_pred = np.argmax(model.predict(X_snr, verbose=0), axis=1)
unique_snrs = np.array(sorted(np.unique(snr_vals)), dtype=int)

overall_acc = []
per_mod_acc = np.full((len(snr_target_names), len(unique_snrs)), np.nan, dtype=np.float32)
for j, s in enumerate(unique_snrs):
    idx = np.where(snr_vals == s)[0]
    overall_acc.append(float(np.mean(y_snr_pred[idx] == y_snr[idx])) * 100.0)

for c in range(len(snr_target_names)):
    cls_mask = y_snr == c
    for j, s in enumerate(unique_snrs):
        idx = np.where((snr_vals == s) & cls_mask)[0]
        if len(idx) > 0:
            per_mod_acc[c, j] = float(np.mean(y_snr_pred[idx] == y_snr[idx])) * 100.0

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].plot(unique_snrs, overall_acc, marker='o', color='blue')
axes[0].set_title('Recognition Accuracy vs. SNR (RML2018)')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].grid(True, alpha=0.4)

for i, mod in enumerate(snr_target_names):
    axes[1].plot(unique_snrs, per_mod_acc[i], marker='o', linewidth=1.0, label=mod)
axes[1].set_title('Accuracy vs. SNR per Modulation Type (RML2018)')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
plt.tight_layout()

png = outputs_dir_50 / '50_rml2018_accuracy_vs_snr_line_plots.png'
plt.savefig(png, dpi=180)
print('Saved line charts:', png)
_save_current_figure("cell_03_figure_09.png")

# %% Cell 4
# Cell 4 : TEST - Noisy Drone RF v2 VGG full-complex eval-only
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

notebook_dir = Path().resolve()
if notebook_dir.name == 'pipelines':
    project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir
elif (notebook_dir / 'pipelines').exists() and (notebook_dir / 'src').exists():
    project_root = notebook_dir
else:
    project_root = notebook_dir

outputs_dir_50 = project_root / 'outputs' / '50_evaluation_comparison'
outputs_dir_50.mkdir(parents=True, exist_ok=True)
noisy_eval_metrics_path = outputs_dir_50 / '50_noisy_drone_rf_v2_eval_metrics.json'
noisy_eval_report_path = outputs_dir_50 / '50_noisy_drone_rf_v2_classification_report.csv'

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    dcfg = cfg.get('datasets', {}).get('noisy_drone_rf_v2', {}) or {}
    data_dir = Path(dcfg.get('data_dir', '/scratch/rameyjm7/datasets/NoisyDroneRFv2'))
else:
    data_dir = Path('/scratch/rameyjm7/datasets/NoisyDroneRFv2')

model_dir = project_root / 'models' / 'noisy_drone_rf_v2'
noisy_model_best_path = model_dir / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras'
noisy_model_final_path = model_dir / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_final.keras'
noisy_model_path = noisy_model_final_path if noisy_model_final_path.exists() else noisy_model_best_path

RANDOM_STATE = 1961
MAX_IQ_SAMPLES = int(os.getenv('NOISY_DRONE_MAX_IQ_SAMPLES', '1048576'))
MIN_SNR_DB = float(os.getenv('NOISY_DRONE_MIN_SNR_DB', '-6'))
DATA_FRACTION = float(os.getenv('NOISY_DRONE_DATA_FRACTION', '0.25'))
BATCH_SIZE = int(os.getenv('NOISY_DRONE_BATCH_SIZE', '8'))
SPEC_NFFT = int(os.getenv('NOISY_DRONE_SPEC_NFFT', '1024'))
SPEC_HOP = int(os.getenv('NOISY_DRONE_SPEC_HOP', '1024'))
SPEC_TIME_BINS = int(os.getenv('NOISY_DRONE_SPEC_TIME_BINS', '1024'))
SPEC_EVAL_WINDOWS = int(os.getenv('NOISY_DRONE_SPEC_EVAL_WINDOWS', '1'))
BURST_SMOOTH_SAMPLES = int(os.getenv('NOISY_DRONE_BURST_SMOOTH_SAMPLES', '512'))
BALANCED_EVAL = os.getenv('NOISY_DRONE_BALANCED_EVAL', '1').lower() not in {'0', 'false', 'no'}
cache_dir = Path(os.getenv(
    'NOISY_DRONE_SPEC_CACHE_DIR',
    '/scratch/rameyjm7/ML-wireless-signal-classification/cache/noisy_drone_rf_v2/spectrogram_full_complex_cache',
))
cache_dir.mkdir(parents=True, exist_ok=True)

print('Noisy Drone RF v2 data:', data_dir)
print('Noisy Drone RF v2 model:', noisy_model_path)
assert data_dir.exists(), f'Missing NoisyDroneRFv2 directory: {data_dir}'
assert noisy_model_path.exists(), f'Missing model: {noisy_model_path}'

sample_re = re.compile(r'IQdata_sample(?P<sample>\d+)_target(?P<target>-?\d+)_snr(?P<snr>-?\d+)\.pt$')
pt_files = sorted(data_dir.rglob('IQdata_sample*_target*_snr*.pt'))
assert pt_files, f'No matching .pt files found under {data_dir}'

rows = []
for filepath in pt_files:
    match = sample_re.search(filepath.name)
    if match:
        rows.append({
            'filepath': str(filepath),
            'sample_id': int(match.group('sample')),
            'target_raw': int(match.group('target')),
            'snr': int(match.group('snr')),
        })

data_df = pd.DataFrame(rows).sort_values('sample_id').reset_index(drop=True)
full_sample_count = len(data_df)
if MIN_SNR_DB > -999:
    data_df = data_df[data_df['snr'] >= MIN_SNR_DB].reset_index(drop=True)
if DATA_FRACTION < 1.0:
    data_df = (
        data_df.groupby('target_raw', group_keys=False)
        .sample(frac=DATA_FRACTION, random_state=RANDOM_STATE)
        .sort_values('sample_id')
        .reset_index(drop=True)
    )

classes = np.array(sorted(data_df['target_raw'].unique()), dtype=np.int64)
class_to_index = {int(c): idx for idx, c in enumerate(classes)}
class_stats_path = data_dir / 'class_stats.csv'
class_stats = pd.read_csv(class_stats_path) if class_stats_path.exists() else None
if class_stats is not None and {'class_int', 'class'}.issubset(class_stats.columns):
    class_name_lookup = dict(zip(class_stats['class_int'].astype(int), class_stats['class'].astype(str)))
else:
    class_name_lookup = {int(c): f'target_{int(c)}' for c in classes}
label_names = [class_name_lookup.get(int(c), f'target_{int(c)}') for c in classes]
data_df['label_idx'] = data_df['target_raw'].map(class_to_index).astype(np.int64)


def load_pt_iq(filepath):
    obj = torch.load(filepath, map_location='cpu')

    def extract_iq(value):
        if isinstance(value, dict):
            preferred_keys = ('x_iq', 'iq', 'IQ', 'x', 'X', 'data', 'samples', 'signal', 'waveform', 'input', 'features', 'arr', 'array')
            for key in preferred_keys:
                if key in value:
                    return extract_iq(value[key])
            for item in value.values():
                if hasattr(item, 'detach') or isinstance(item, (np.ndarray, list, tuple, dict)):
                    try:
                        return extract_iq(item)
                    except (TypeError, ValueError, KeyError):
                        continue
            raise KeyError(f'No IQ-like payload found in {filepath}; keys={list(value.keys())}')
        if isinstance(value, (tuple, list)):
            if not value:
                raise ValueError(f'Empty sequence payload in {filepath}')
            return extract_iq(value[0])
        return value

    obj = extract_iq(obj)
    arr = obj.detach().cpu().numpy() if hasattr(obj, 'detach') else np.asarray(obj)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if np.iscomplexobj(arr):
            arr = np.stack([arr.real, arr.imag], axis=-1)
        else:
            if arr.size % 2 != 0:
                arr = arr[:-1]
            arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        elif arr.shape[-1] != 2:
            raise ValueError(f'Expected IQ tensor with final dimension 2, got {arr.shape} in {filepath}')
    else:
        if arr.shape[-1] == 2:
            arr = arr.reshape(-1, 2)
        elif arr.shape[0] == 2:
            arr = np.moveaxis(arr, 0, -1).reshape(-1, 2)
        else:
            raise ValueError(f'Expected IQ tensor with two channels, got {arr.shape} in {filepath}')
    return np.asarray(arr, dtype=np.float32)

raw_sample_len = load_pt_iq(data_df.iloc[0]['filepath']).shape[0]
SAMPLE_LEN = min(raw_sample_len, MAX_IQ_SAMPLES)
num_classes = len(classes)
input_shape = (SPEC_NFFT, SPEC_TIME_BINS, 2)


def find_burst_start(x, window_len):
    if x.shape[0] <= window_len:
        return 0
    power = np.mean(np.square(x.astype(np.float32)), axis=1)
    smooth_len = max(1, min(BURST_SMOOTH_SAMPLES, power.shape[0] // 8))
    if smooth_len > 1:
        kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
        power = np.convolve(power, kernel, mode='same')
    center = int(np.argmax(power))
    return int(np.clip(center - window_len // 2, 0, x.shape[0] - window_len))


def normalize_iq_window(iq):
    iq = np.asarray(iq[:, :2], dtype=np.float32)
    iq = iq - np.mean(iq, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(iq)) + 1e-8)
    return iq / scale


def resize_time_axis(arr, target_bins):
    if arr.shape[1] == target_bins:
        return arr
    src_x = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    return np.stack([np.interp(dst_x, src_x, row).astype(np.float32) for row in arr], axis=0)


def iq_window_to_spectrogram(iq):
    iq = normalize_iq_window(iq[:, :2])
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    if len(complex_iq) < SPEC_NFFT:
        complex_iq = np.pad(complex_iq, (0, SPEC_NFFT - len(complex_iq)), mode='constant')
    starts = np.arange(0, len(complex_iq) - SPEC_NFFT + 1, SPEC_HOP)
    if starts.size == 0:
        starts = np.array([0])
    window = np.hanning(SPEC_NFFT).astype(np.float32)
    frames = np.stack([complex_iq[s:s + SPEC_NFFT] * window for s in starts], axis=0)
    fft_complex = np.fft.fftshift(np.fft.fft(frames, n=SPEC_NFFT, axis=1), axes=1).T / float(SPEC_NFFT)
    real_part = resize_time_axis(fft_complex.real.astype(np.float32), SPEC_TIME_BINS)
    imag_part = resize_time_axis(fft_complex.imag.astype(np.float32), SPEC_TIME_BINS)
    spec = np.stack([real_part, imag_part], axis=-1).astype(np.float32)
    spec = spec / (np.std(spec) + 1e-6)
    return np.clip(spec, -6.0, 6.0).astype(np.float32)


def spectrogram_cache_path(filepath, snr_value):
    src = Path(str(filepath))
    name = f'{src.stem}_full_complex_len{SAMPLE_LEN}_nfft{SPEC_NFFT}_hop{SPEC_HOP}_tb{SPEC_TIME_BINS}_snr{int(float(snr_value))}.npz'
    return cache_dir / name


def safe_load_cached_spectrogram(cache_path):
    try:
        with np.load(cache_path) as data:
            x = data['x'].astype(np.float32)
        if x.shape != input_shape or not np.isfinite(x).all():
            raise ValueError(f'bad cached spectrogram: {x.shape}')
        return x
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile):
        Path(cache_path).unlink(missing_ok=True)
        return None


def write_spectrogram_cache_atomic(cache_path, x):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f'.{os.getpid()}.tmp')
    with tmp_path.open('wb') as handle:
        np.savez_compressed(handle, x=x.astype(np.float32))
    tmp_path.replace(cache_path)


def prepare_spectrogram(filepath, snr_value):
    cache_path = spectrogram_cache_path(filepath, snr_value)
    if cache_path.exists():
        cached = safe_load_cached_spectrogram(cache_path)
        if cached is not None:
            return cached
    x_full = load_pt_iq(filepath)
    if x_full.shape[0] < SAMPLE_LEN:
        x_full = np.pad(x_full, ((0, SAMPLE_LEN - x_full.shape[0]), (0, 0)), mode='constant')
    start = find_burst_start(x_full, SAMPLE_LEN)
    x = iq_window_to_spectrogram(x_full[start:start + SAMPLE_LEN, :2])
    write_spectrogram_cache_atomic(cache_path, x)
    return x.astype(np.float32)


def prepare_spectrogram_windows(filepath, snr_value, n_windows=None):
    n_windows = SPEC_EVAL_WINDOWS if n_windows is None else int(n_windows)
    if n_windows <= 1:
        return prepare_spectrogram(filepath, snr_value)[None, ...]
    x_full = load_pt_iq(filepath)
    if x_full.shape[0] < SAMPLE_LEN:
        x_full = np.pad(x_full, ((0, SAMPLE_LEN - x_full.shape[0]), (0, 0)), mode='constant')
    center_start = find_burst_start(x_full, SAMPLE_LEN)
    if x_full.shape[0] <= SAMPLE_LEN:
        starts = [0]
    else:
        stride = max(1, SAMPLE_LEN // max(4, n_windows + 1))
        offsets = (np.arange(n_windows) - ((n_windows - 1) / 2.0)) * stride
        starts = sorted(set(int(np.clip(center_start + off, 0, x_full.shape[0] - SAMPLE_LEN)) for off in offsets))
    return np.stack([iq_window_to_spectrogram(x_full[start:start + SAMPLE_LEN, :2]) for start in starts], axis=0).astype(np.float32)


def make_balanced_eval_df(split_df):
    target_n = int(split_df['label_idx'].value_counts().min())
    return split_df.groupby('label_idx', group_keys=False).sample(n=target_n, random_state=RANDOM_STATE).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

idx = np.arange(len(data_df))
train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=data_df['label_idx'])
test_df = data_df.iloc[test_idx].reset_index(drop=True)
balanced_test_df = make_balanced_eval_df(test_df) if BALANCED_EVAL else None

noisy_model = load_model(noisy_model_path, compile=False)
print('Loaded noisy drone model:', noisy_model_path)


def predict_noisy(split_df, name):
    probs = []
    total = len(split_df)
    for i, row in enumerate(split_df.itertuples(index=False), start=1):
        x = prepare_spectrogram_windows(row.filepath, row.snr, n_windows=SPEC_EVAL_WINDOWS)
        probs.append(noisy_model.predict(x, batch_size=BATCH_SIZE, verbose=0).mean(axis=0))
        if i <= 3 or i % 100 == 0 or i == total:
            print(f'{name}: {i}/{total}', flush=True)
    return split_df.copy(), np.asarray(probs, dtype=np.float32)

natural_df, natural_probs = predict_noisy(test_df, 'noisy-natural')
y_true = natural_df['label_idx'].to_numpy(dtype=np.int64)
y_pred = natural_probs.argmax(axis=1)
natural_metrics = {
    'accuracy': float(accuracy_score(y_true, y_pred)),
    'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
}
print('Noisy Drone RF v2 natural metrics:', natural_metrics)
print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
pd.DataFrame(classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)).transpose().to_csv(noisy_eval_report_path)

metrics = {
    'model': 'noisy_drone_rf_v2_vgg_full_complex_spectrogram',
    'model_path': str(noisy_model_path),
    'eval_windows': int(SPEC_EVAL_WINDOWS),
    'sample_len': int(SAMPLE_LEN),
    'min_snr_db': float(MIN_SNR_DB),
    'data_fraction': float(DATA_FRACTION),
    'num_classes': int(num_classes),
    'test_samples': int(len(test_df)),
    'label_names': label_names,
    **natural_metrics,
}

if balanced_test_df is not None:
    balanced_df, balanced_probs = predict_noisy(balanced_test_df, 'noisy-balanced')
    y_balanced_true = balanced_df['label_idx'].to_numpy(dtype=np.int64)
    y_balanced_pred = balanced_probs.argmax(axis=1)
    metrics.update({
        'balanced_accuracy': float(accuracy_score(y_balanced_true, y_balanced_pred)),
        'balanced_macro_f1': float(f1_score(y_balanced_true, y_balanced_pred, average='macro', zero_division=0)),
        'balanced_weighted_f1': float(f1_score(y_balanced_true, y_balanced_pred, average='weighted', zero_division=0)),
        'balanced_test_samples': int(len(balanced_test_df)),
    })
    print('Noisy Drone RF v2 balanced metrics:', {k: metrics[k] for k in metrics if k.startswith('balanced_')})

noisy_eval_metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('Saved noisy drone eval metrics:', noisy_eval_metrics_path)
print('Saved noisy drone report:', noisy_eval_report_path)

# %% Cell 5
# Cell 5 : TEST - Final cross-dataset model comparison
from pathlib import Path
import json
import numpy as np
import pandas as pd
import importlib
import matplotlib as mpl
if not hasattr(mpl, 'backends'):
    mpl.backends = importlib.import_module('matplotlib.backends')
import matplotlib.pyplot as plt
plt.close('all')

notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir
outputs_dir_50 = project_root / 'outputs' / '50_evaluation_comparison'
outputs_dir_50.mkdir(parents=True, exist_ok=True)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f'Could not read {path}: {exc}')
        return None


def add_row(
    rows,
    dataset,
    model_name,
    model_family,
    source,
    split='all_test',
    accuracy=None,
    macro_f1=None,
    weighted_f1=None,
    model_path=None,
    notes='',
):
    model_path = Path(model_path) if model_path is not None else None
    rows.append({
        'dataset': dataset,
        'model_name': model_name,
        'model_family': model_family,
        'split': split,
        'eval_accuracy': accuracy,
        'eval_macro_f1': macro_f1,
        'eval_weighted_f1': weighted_f1,
        'model_path': str(model_path) if model_path is not None else '',
        'model_exists': bool(model_path.exists()) if model_path is not None else None,
        'model_size_mb': round(model_path.stat().st_size / (1024 * 1024), 3) if model_path is not None and model_path.exists() else None,
        'source': str(source),
        'notes': notes,
    })

rows = []

# RML2016 comparison artifacts from notebook 33.
rml2016_path = project_root / 'outputs' / 'rml2016' / 'rml2016_cnn_transformer_comparison.json'
rml2016_rows = read_json(rml2016_path)
if isinstance(rml2016_rows, list):
    for item in rml2016_rows:
        add_row(
            rows,
            dataset='rml2016',
            model_name=item.get('model', 'unknown'),
            model_family=item.get('model', 'unknown'),
            split=item.get('split', 'unknown'),
            accuracy=item.get('accuracy'),
            macro_f1=item.get('macro_f1'),
            weighted_f1=item.get('weighted_f1'),
            source=rml2016_path,
            notes='Pipeline 33 comparison artifact',
        )

# RML2018 best checkpoint artifact from notebook 31/41.
rml2018_path = project_root / 'outputs' / 'rml2018' / 'rml2018_checkpoint_metrics.json'
rml2018_rows = read_json(rml2018_path)
if isinstance(rml2018_rows, list) and rml2018_rows:
    best = max(rml2018_rows, key=lambda row: row.get('eval_accuracy', float('-inf')))
    add_row(
        rows,
        dataset='rml2018',
        model_name=Path(best.get('checkpoint', 'rml2018_lstm_rnn')).stem,
        model_family='lstm',
        split='all_test',
        accuracy=best.get('eval_accuracy'),
        macro_f1=None,
        weighted_f1=None,
        source=rml2018_path,
        notes='Best checkpoint by eval_accuracy',
    )

# DeepRadar2022 artifact from CNN-transformer experiments if present.
deepradar_path = project_root / 'outputs' / 'deepradar2022' / 'deepradar2022_cnn_transformer_stage2_result.json'
deepradar = read_json(deepradar_path)
if isinstance(deepradar, dict):
    add_row(
        rows,
        dataset='deepradar2022',
        model_name=deepradar.get('model', 'deepradar2022_cnn_transformer'),
        model_family='cnn_transformer',
        split=deepradar.get('split', 'all_test'),
        accuracy=deepradar.get('accuracy'),
        macro_f1=deepradar.get('macro_f1'),
        weighted_f1=deepradar.get('weighted_f1'),
        source=deepradar_path,
        notes='Pipeline 33 DeepRadar artifact',
    )

# Noisy Drone RF v2: prefer Cell 4 eval-only metrics generated in this notebook;
# fall back to notebook 44/33 metrics if Cell 4 has not been run yet.
noisy_model_best_path = project_root / 'models' / 'noisy_drone_rf_v2' / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras'
noisy_model_final_path = project_root / 'models' / 'noisy_drone_rf_v2' / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_final.keras'
noisy_model_path = noisy_model_final_path if noisy_model_final_path.exists() else noisy_model_best_path
noisy_50_path = outputs_dir_50 / '50_noisy_drone_rf_v2_eval_metrics.json'
noisy_44_path = project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_metrics.json'
noisy_33_path = project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '33_noisy_drone_rf_v2_vgg_full_complex_spectrogram_metrics.json'
noisy_path = noisy_50_path if noisy_50_path.exists() else (noisy_44_path if noisy_44_path.exists() else noisy_33_path)
noisy = read_json(noisy_path)
if isinstance(noisy, dict):
    model_name = noisy.get('model', 'noisy_drone_rf_v2_vgg_full_complex_spectrogram')
    metric_source_note = 'Pipeline 50 Cell 4 eval-only metrics' if noisy_path == noisy_50_path else ('Pipeline 44 eval metrics fallback' if noisy_path == noisy_44_path else 'Pipeline 33 fallback metrics; run Cell 4 to refresh')
    add_row(
        rows,
        dataset='noisy_drone_rf_v2',
        model_name=model_name,
        model_family='vgg_full_complex_spectrogram',
        split='natural_test',
        accuracy=noisy.get('natural_accuracy', noisy.get('accuracy')),
        macro_f1=noisy.get('natural_macro_f1', noisy.get('macro_f1')),
        weighted_f1=noisy.get('natural_weighted_f1', noisy.get('weighted_f1')),
        model_path=noisy_model_path,
        source=noisy_path,
        notes=metric_source_note,
    )
    if 'balanced_accuracy' in noisy:
        add_row(
            rows,
            dataset='noisy_drone_rf_v2',
            model_name=model_name,
            model_family='vgg_full_complex_spectrogram',
            split='balanced_test',
            accuracy=noisy.get('balanced_accuracy'),
            macro_f1=noisy.get('balanced_macro_f1'),
            weighted_f1=noisy.get('balanced_weighted_f1'),
            model_path=noisy_model_path,
            source=noisy_path,
            notes=metric_source_note,
        )
elif noisy_model_path.exists():
    add_row(
        rows,
        dataset='noisy_drone_rf_v2',
        model_name='noisy_drone_rf_v2_vgg_full_complex_spectrogram',
        model_family='vgg_full_complex_spectrogram',
        split='not_evaluated',
        model_path=noisy_model_path,
        source=noisy_model_path,
        notes='Canonical VGG model exists; run notebook 44 to generate metrics',
    )

comparison_df = pd.DataFrame(rows)
if comparison_df.empty:
    raise FileNotFoundError('No comparison metrics found. Run/evaluate pipelines 33/41/44 first.')

for col in ['eval_accuracy', 'eval_macro_f1', 'eval_weighted_f1']:
    comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')

comparison_df = comparison_df.sort_values(['dataset', 'split', 'eval_accuracy'], ascending=[True, True, False]).reset_index(drop=True)
comparison_csv = outputs_dir_50 / '50_cross_dataset_model_comparison.csv'
comparison_json = outputs_dir_50 / '50_cross_dataset_model_comparison.json'
comparison_df.to_csv(comparison_csv, index=False)
comparison_json.write_text(comparison_df.to_json(orient='records', indent=2), encoding='utf-8')
print('Saved comparison CSV:', comparison_csv)
print('Saved comparison JSON:', comparison_json)
print(comparison_df)

plot_df = comparison_df.dropna(subset=['eval_accuracy']).copy()
plot_df['label'] = plot_df['dataset'] + '\n' + plot_df['model_name'] + '\n' + plot_df['split']
plot_df = plot_df.sort_values('eval_accuracy')

fig_height = max(5, 0.5 * len(plot_df))
fig, ax = plt.subplots(figsize=(12, fig_height))
ax.barh(plot_df['label'], plot_df['eval_accuracy'] * 100.0)
ax.set_xlabel('Evaluation accuracy (%)')
ax.set_title('Cross-Dataset Model Comparison')
ax.grid(True, axis='x', alpha=0.3)
for idx, value in enumerate(plot_df['eval_accuracy'] * 100.0):
    ax.text(value + 0.5, idx, f'{value:.2f}%', va='center', fontsize=9)
plt.tight_layout()
plot_path = outputs_dir_50 / '50_cross_dataset_model_comparison_accuracy.png'
plt.savefig(plot_path, dpi=180)
print('Saved comparison plot:', plot_path)
_save_current_figure("cell_05_figure_10.png")



# Report final standalone figure artifacts instead of compositing them into one image.
standalone_artifacts = {
    'cross_dataset_accuracy_summary': outputs_dir_50 / '50_cross_dataset_model_comparison_accuracy.png',
    'noisy_drone_balanced_confusion_matrix': project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_balanced_confusion_matrix.png',
    'noisy_drone_accuracy_vs_snr': project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr.png',
    'noisy_drone_accuracy_vs_snr_per_class': project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '44_noisy_drone_rf_v2_vgg_full_complex_accuracy_vs_snr_per_class.png',
}

artifact_rows = []
for name, path in standalone_artifacts.items():
    artifact_rows.append({
        'artifact': name,
        'path': str(path),
        'exists': path.exists(),
    })
artifact_df = pd.DataFrame(artifact_rows)
artifact_csv = outputs_dir_50 / '50_final_standalone_figure_artifacts.csv'
artifact_df.to_csv(artifact_csv, index=False)
print('Saved standalone artifact index:', artifact_csv)
print(artifact_df)

missing = artifact_df[~artifact_df['exists']]
if not missing.empty:
    print('Missing artifacts. Run notebook 44 Cells 3-4 and this Cell 4 again if needed.')
else:
    from matplotlib.image import imread

    for name, path in standalone_artifacts.items():
        img = imread(path)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(name.replace('_', ' ').title(), fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        _save_current_figure("cell_05_figure_11.png")
    print('Standalone final figures displayed separately; no composite image generated.')
