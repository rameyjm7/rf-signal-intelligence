#!/usr/bin/env python3
"""Pipeline converted from the legacy 32_lstm_deepradar2022 workflow."""

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
# Cell 1 : Train the DeepRadar2022 CNN-BiLSTM model
# https://www.kaggle.com/jacobramey
# https://github.com/rameyjm7

import numpy as np
import h5py, scipy.io as sio, os, sklearn
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from rf_signal_intelligence.features.spectrogram import normalize_iq_per_sample, append_snr_feature
from rf_signal_intelligence.models.deepradar_cnn_bilstm import build_deepradar_cnn_bilstm
import tensorflow as tf
import os
from pathlib import Path
import yaml

# ------------------------------
# Load DeepRadar2022 dataset (local)
# ------------------------------
notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir
cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text())
    path = Path(local_cfg.get('dataset_root', '/scratch/rameyjm7/datasets')) / 'DeepRadar2022'
else:
    path = Path('/scratch/rameyjm7/datasets/DeepRadar2022')

print('Loading DeepRadar2022 from:', path)

def load_h5(filepath, key):
    with h5py.File(filepath, "r") as f:
        return np.array(f[key], dtype="float32").T

def load_mat(filepath, key):
    d = sio.loadmat(filepath)
    return d[key]

# Core signal data
X_train = load_h5(path / "X_train.mat", "X_train")
X_val   = load_h5(path / "X_val.mat", "X_val")
X_test  = load_h5(path / "X_test.mat", "X_test")

# Label matrices and metadata
Y_train = load_mat(path / "Y_train.mat", "Y_train")
Y_val   = load_mat(path / "Y_val.mat", "Y_val")
Y_test  = load_mat(path / "Y_test.mat", "Y_test")

lbl_train = load_mat(path / "lbl_train.mat", "lbl_train")
lbl_val   = load_mat(path / "lbl_val.mat", "lbl_val")
lbl_test  = load_mat(path / "lbl_test.mat", "lbl_test")

print("✅ DeepRadar2022 dataset successfully loaded from local path.")


# Extract modulation class and SNR
cls_train, snr_train = lbl_train[:,0].astype(int)-1, lbl_train[:,1]
cls_val,   snr_val   = lbl_val[:,0].astype(int)-1,   lbl_val[:,1]
cls_test,  snr_test  = lbl_test[:,0].astype(int)-1,  lbl_test[:,1]

# ------------------------------
# Filter training/validation by SNR ≥ -6 dB
# ------------------------------
train_mask = snr_train >= -6
val_mask   = snr_val >= -6
X_train, Y_train, snr_train = X_train[train_mask], Y_train[train_mask], snr_train[train_mask]
X_val,   Y_val,   snr_val   = X_val[val_mask],   Y_val[val_mask],   snr_val[val_mask]

print(f"Training samples kept: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ------------------------------
# Shuffle
# ------------------------------
np.random.seed(1961)
X_train, Y_train, snr_train = sklearn.utils.shuffle(X_train, Y_train, snr_train, random_state=1961)
X_val,   Y_val,   snr_val   = sklearn.utils.shuffle(X_val, Y_val, snr_val, random_state=1961)
X_test,  Y_test,  snr_test  = sklearn.utils.shuffle(X_test, Y_test, snr_test, random_state=1961)

# ------------------------------
# Normalize IQ per sample
# ------------------------------
# Implemented in rf_signal_intelligence.features.spectrogram.normalize_iq_per_sample
X_train = normalize_iq_per_sample(X_train)
X_val   = normalize_iq_per_sample(X_val)
X_test  = normalize_iq_per_sample(X_test)

# ------------------------------
# Append SNR as a third channel
# ------------------------------
# Implemented in rf_signal_intelligence.features.spectrogram.append_snr_feature
X_train = append_snr_feature(X_train, snr_train)
X_val   = append_snr_feature(X_val, snr_val)
X_test  = append_snr_feature(X_test, snr_test)

input_shape = (1024, 3)
num_classes = 23

# ------------------------------
# Enable mixed precision
# ------------------------------
mixed_precision.set_global_policy("mixed_float16")

# ------------------------------
# Build CNN + Bidirectional LSTM model
# ------------------------------
# Architecture implemented in rf_signal_intelligence.models.deepradar_cnn_bilstm.build_deepradar_cnn_bilstm
model = build_deepradar_cnn_bilstm(input_shape, num_classes)
model.summary()

# ------------------------------
# Training setup
# ------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint("deepradar2022_cnn_bilstm_highsnr.keras", save_best_only=True)
]

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=35,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------
# Evaluate on full test set (all SNRs)
# ------------------------------
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy (all SNRs): {acc*100:.2f}%")

Y_pred = model.predict(X_test, verbose=0)
y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix (CNN + BiLSTM, Trained ≥ –6 dB, Evaluated All)")
_save_current_figure("cell_01_figure_01.png")

print("Classification Report:")
print(classification_report(y_true, y_pred))

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
plt.title("Recognition Accuracy vs SNR (CNN + BiLSTM, Trained ≥ –6 dB)")
plt.grid(True)
_save_current_figure("cell_01_figure_02.png")

# %% Cell 2
# Cell 2 : Save the final DeepRadar2022 model
# ------------------------------
# Save final model in .keras format and plot labeled confusion matrices
# ------------------------------
# Define model path
model_path = os.path.join("..", "models", "deepradar2022", "deepradar2022_cnn_bilstm_final.keras")

# Save model architecture, weights, and optimizer state in one file
model.save(model_path, include_optimizer=True)
print(f"Model saved successfully at: {os.path.abspath(model_path)}")

# %% Cell 3
# Cell 3 : Evaluate the saved DeepRadar2022 model
# ================================================
# DeepRadar2022 – Full Evaluation (Standalone Cell)
# ================================================
import os
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from rf_signal_intelligence.features.spectrogram import normalize_iq_per_sample, append_snr_feature

# ---------------------------------------------------------
# Resolve project paths (safe for pipelines anywhere)
# ---------------------------------------------------------
notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text())
    data_dir = Path(local_cfg.get('dataset_root', '/scratch/rameyjm7/datasets')) / 'DeepRadar2022'
else:
    data_dir = Path('/scratch/rameyjm7/datasets/DeepRadar2022')
model_path = project_root / "models" / "deepradar2022" / "deepradar2022_cnn_bilstm_final.keras"

print("Pipeline directory:", notebook_dir)
print("Project root:", project_root)
print("Dataset directory:", data_dir)
print("Model path:", model_path)

assert data_dir.exists(), f"DeepRadar2022 directory not found: {data_dir}"
assert model_path.exists(), f"Model file missing: {model_path}"

# ---------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------
def load_h5(filepath, key):
    with h5py.File(filepath, "r") as f:
        return np.array(f[key], dtype="float32").T

def load_mat(filepath, key):
    d = sio.loadmat(filepath)
    return d[key]

# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------
print("Loading DeepRadar2022 dataset...")
X_test  = load_h5(data_dir / "X_test.mat",  "X_test")
Y_test  = load_mat(data_dir / "Y_test.mat", "Y_test")
lbl_test = load_mat(data_dir / "lbl_test.mat", "lbl_test")  # columns: [class_id, SNR]

cls_test = lbl_test[:, 0].astype(int) - 1
snr_test = lbl_test[:, 1]

print("Loaded shapes:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
print("Labels:", lbl_test.shape)

# ---------------------------------------------------------
# Normalize IQ per sample
# ---------------------------------------------------------
# Implemented in rf_signal_intelligence.features.spectrogram.normalize_iq_per_sample
X_test = normalize_iq_per_sample(X_test)

# ---------------------------------------------------------
# Append SNR as a 3rd channel
# ---------------------------------------------------------
# Implemented in rf_signal_intelligence.features.spectrogram.append_snr_feature
X_test = append_snr_feature(X_test, snr_test)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
print("Loading trained CNN+BiLSTM model...")
model = load_model(model_path)
print("Model loaded.")

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest accuracy (all SNRs): {acc*100:.2f}%")

# Predict
Y_pred = model.predict(X_test, verbose=0)
y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(Y_pred, axis=1)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------
mod_labels = [
    "LFM", "2FSK", "4FSK", "8FSK", "Costas",
    "2PSK", "4PSK", "8PSK",
    "Barker", "Huffman", "Frank",
    "P1", "P2", "P3", "P4", "Px",
    "Zadoff-Chu",
    "T1", "T2", "T3", "T4",
    "NM", "Noise"
]

# ---------------------------------------------------------
# Confusion Matrix – All SNRs
# ---------------------------------------------------------
cm_all = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(13, 11))
sns.heatmap(
    cm_all, cmap="Blues", annot=False,
    xticklabels=mod_labels, yticklabels=mod_labels
)
plt.title("Confusion Matrix – All SNRs")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
_save_current_figure("cell_03_figure_03.png")

# ---------------------------------------------------------
# Confusion Matrix – High SNR (>= -6 dB)
# ---------------------------------------------------------
high_mask = snr_test >= -6
y_true_high = y_true[high_mask]
y_pred_high = y_pred[high_mask]

cm_high = confusion_matrix(y_true_high, y_pred_high)
plt.figure(figsize=(13, 11))
sns.heatmap(
    cm_high, cmap="Greens", annot=False,
    xticklabels=mod_labels, yticklabels=mod_labels
)
plt.title("Confusion Matrix – SNR ≥ -6 dB")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
_save_current_figure("cell_03_figure_04.png")

# ---------------------------------------------------------
# Reports
# ---------------------------------------------------------
print("\nClassification Report – All SNRs:")
print(classification_report(y_true, y_pred, target_names=mod_labels, digits=3))

print("\nClassification Report – SNR ≥ -6 dB:")
print(classification_report(y_true_high, y_pred_high, target_names=mod_labels, digits=3))
