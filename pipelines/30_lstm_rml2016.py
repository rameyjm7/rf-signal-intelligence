#!/usr/bin/env python3
"""Pipeline converted from the legacy 30_lstm_rml2016 workflow."""

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
# Cell 1 : Train and evaluate the RML2016 LSTM model
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Resolve paths when run from either the repo root or the pipelines directory.
project_root = Path.cwd().resolve()
if project_root.name == "pipelines":
    project_root = project_root.parent
elif not (project_root / "pyproject.toml").exists() and (project_root.parent / "pyproject.toml").exists():
    project_root = project_root.parent

src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from rf_signal_intelligence.data.rml import load_rml2016_pickle, rml2016_arrays
from rf_signal_intelligence.models.rml2016_lstm import build_rml2016_lstm_model

model_dir = project_root / "models" / "rml2016"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "rml2016_lstm_rnn_2024.keras"

outputs_dir = project_root / "outputs" / "rml2016"
outputs_dir.mkdir(parents=True, exist_ok=True)

TRAIN_MODEL = os.environ.get("RFSI_RML2016_TRAIN", "1") != "0"
RESUME_FROM_EXISTING = os.environ.get("RFSI_RML2016_RESUME", "1") != "0"
EPOCHS = int(os.environ.get("RFSI_RML2016_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("RFSI_RML2016_BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("RFSI_RML2016_LR", "1e-4"))

print("Resolved model path:", model_path)
print("Training enabled:", TRAIN_MODEL)
print("Resume from existing model:", RESUME_FROM_EXISTING)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)

# Resolve dataset path from local config (written by 10_download_data.py)
cfg_path = project_root / "configs" / "local_data_paths.yaml"
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text()) or {}
    rml2016_pkl = local_cfg.get("datasets", {}).get(
        "rml2016", {}
    ).get("pkl", "/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl")
else:
    rml2016_pkl = "/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl"

rml2016_pkl = Path(rml2016_pkl)
print("Resolved RML2016 data path:", rml2016_pkl)
assert rml2016_pkl.exists(), f"Dataset not found: {rml2016_pkl}"

# Load the data
data = load_rml2016_pickle(rml2016_pkl)

# Prepare the data using the notebook-30 RML2016 format: I, Q, and constant SNR channel.
def prepare_data(data):
    X, y_text, _class_names = rml2016_arrays(data)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )
    return X_train, X_test, y_train, y_test, label_encoder

# Prepare the data
X_train, X_test, y_train, y_test, label_encoder = prepare_data(data)
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(label_encoder.classes_)
print("Input shape:", input_shape)
print("Classes:", list(label_encoder.classes_))

if RESUME_FROM_EXISTING and model_path.exists():
    print(f"Loading existing model from {model_path}")
    model = load_model(model_path)
else:
    print("Building a new RML2016 LSTM model")
    model = build_rml2016_lstm_model(input_shape, num_classes, learning_rate=LEARNING_RATE)

if TRAIN_MODEL:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}
    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        np.nan_to_num(X_train, nan=0.0),
        y_train,
        validation_data=(np.nan_to_num(X_test, nan=0.0), y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )
    model.save(model_path)
    print(f"Saved trained model to {model_path}")
else:
    history = None
    print("Skipping training because RFSI_RML2016_TRAIN=0")

# Make predictions on the test set
y_pred = np.argmax(model.predict(X_test, verbose=False), axis=1)

# Plot the confusion matrix for all SNR levels
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Modulation Classification (All SNR Levels)")
_save_current_figure("cell_01_figure_01.png")

# Print the classification report
print("Classification Report for Modulation Types:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot feature importance (if using a tree-based model)
# Plot confusion matrix for SNR > 5 dB subset
snr_above_5_indices = np.where(X_test[:, 0, 2] > 5)  # Assuming SNR values are in the third column
X_test_snr_above_5 = X_test[snr_above_5_indices]
y_test_snr_above_5 = y_test[snr_above_5_indices]

# Make predictions on the SNR > 5 dB subset
y_pred_snr_above_5 = np.argmax(model.predict(X_test_snr_above_5, verbose=False), axis=1)

# Plot confusion matrix for SNR > 5 dB
conf_matrix_snr_above_5 = confusion_matrix(y_test_snr_above_5, y_pred_snr_above_5)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_snr_above_5, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Modulation Classification (SNR > 5 dB)")
_save_current_figure("cell_01_figure_02.png")

# Print the classification report for SNR > 5 dB
print("Classification Report for Modulation Types (SNR > 5 dB):")
print(classification_report(y_test_snr_above_5, y_pred_snr_above_5, target_names=label_encoder.classes_))

# %% Cell 2
# Cell 2 : Plot recognition accuracy by SNR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load data and prepare using your prepare_data function
X_train, X_test, y_train, y_test, label_encoder = prepare_data(data)

# Evaluate accuracy for each SNR level
unique_snrs = sorted(set(X_test[:, 0, -1]))  # SNR values are in the last column of each sample
accuracy_per_snr = []

for snr in unique_snrs:
    # Select samples with the current SNR
    snr_indices = np.where(X_test[:, 0, -1] == snr)
    X_snr = X_test[snr_indices]
    y_snr = y_test[snr_indices]

    # Predict and calculate accuracy
    y_pred = np.argmax(model.predict(X_snr,verbose=0), axis=1)
    accuracy = accuracy_score(y_snr, y_pred)
    accuracy_per_snr.append(accuracy * 100)  # Convert to percentage

    print(f"SNR: {snr} dB, Accuracy: {accuracy * 100:.2f}%")

# Find the peak accuracy and its corresponding SNR
peak_accuracy = max(accuracy_per_snr)
peak_snr = unique_snrs[accuracy_per_snr.index(peak_accuracy)]

# Plot Recognition Accuracy vs. SNR
plt.figure(figsize=(10, 6))
plt.plot(unique_snrs, accuracy_per_snr, 'b-o', label='Recognition Accuracy')
plt.xlabel("SNR (dB)")
plt.ylabel("Recognition Accuracy (%)")
plt.title(f"Recognition Accuracy vs. SNR (Peak Accuracy: {peak_accuracy:.2f}%)")

# Mark the peak accuracy point
plt.plot(peak_snr, peak_accuracy, 'ro')  # Red dot at the peak
plt.text(peak_snr, peak_accuracy + 1, f"{peak_accuracy:.2f}%", 
         ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.legend()
plt.grid(True)
plt.ylim(0, 100)
_save_current_figure("cell_02_figure_03.png")

# %% Cell 3
# Cell 3 : Plot per-modulation accuracy by SNR
import tensorflow as tf
from rf_signal_intelligence.plots import plot_modulation_accuracy_v_snr

# Set TensorFlow logging level to suppress most of the output
tf.get_logger().setLevel('ERROR')

plot_modulation_accuracy_v_snr(model, X_test, y_test, label_encoder)
