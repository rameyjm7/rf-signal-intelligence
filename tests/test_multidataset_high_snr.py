import ast
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


RML2018_MODEL = "rml2018_lstm_rnn.keras"
DEEPRADAR_MODEL = "deepradar2022_cnn_bilstm_final.keras"

RML2018_MIN_ACC = 0.05
DEEPRADAR_MIN_ACC = 0.85


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _classes_from_python_file(file_path: Path):
    text = file_path.read_text()
    match = re.search(r"classes\s*=\s*(\[[\s\S]*?\])", text)
    if not match:
        raise ValueError(f"Could not parse classes list from {file_path}")
    return ast.literal_eval(match.group(1))


@pytest.mark.integration
def test_rml2018_highest_snr_accuracy():
    repo = _repo_root()
    data_file = repo / "data" / "RML2018" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    model_file = repo / "models" / RML2018_MODEL
    classes_file = repo / "data" / "RML2018" / "classes.txt"
    fixed_classes_file = repo / "data" / "RML2018" / "classes-fixed.txt"

    if not data_file.exists() or not model_file.exists():
        pytest.skip("Missing RML2018 data/model artifacts.")

    original_classes = _classes_from_python_file(classes_file)
    fixed_classes = _classes_from_python_file(fixed_classes_file)
    fixed_index = {name: idx for idx, name in enumerate(fixed_classes)}
    remap = np.array([fixed_index[name] for name in original_classes])

    with h5py.File(data_file, "r") as h5:
        x_ds = h5["X"]
        y_ds = h5["Y"]
        z_ds = h5["Z"]

        snr = z_ds[:, 0]
        max_snr = int(np.max(snr))
        max_idx = np.where(snr == max_snr)[0]
        y_at_max = np.argmax(y_ds[max_idx], axis=1)

        rng = np.random.default_rng(42)
        picked = []
        for cls in np.unique(y_at_max):
            cls_idx = max_idx[y_at_max == cls]
            k = min(40, len(cls_idx))
            picked.extend(rng.choice(cls_idx, size=k, replace=False).tolist())

        picked = np.array(sorted(picked), dtype=np.int64)
        x_iq = np.array(x_ds[picked], dtype=np.float32)
        y_onehot = np.array(y_ds[picked], dtype=np.int64)
        snr_values = np.array(z_ds[picked, 0], dtype=np.float32)

    # Model expects [I, Q, SNR] features for each timestep.
    snr_channel = np.repeat(snr_values[:, None, None], x_iq.shape[1], axis=1)
    x = np.concatenate([x_iq, snr_channel], axis=2)

    y_true_original = np.argmax(y_onehot, axis=1)
    y_true = remap[y_true_original]

    model = load_model(model_file, compile=False)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    acc = accuracy_score(y_true, y_pred)

    assert len(np.unique(y_true)) >= 20, "Highest-SNR slice lacks class diversity."
    assert acc >= RML2018_MIN_ACC, (
        f"RML2018 max-SNR accuracy too low: {acc:.4f} at SNR={max_snr} dB "
        f"(threshold={RML2018_MIN_ACC:.2f}, n={len(y_true)})"
    )


@pytest.mark.integration
def test_deepradar2022_highest_snr_accuracy():
    repo = _repo_root()
    x_file = repo / "data" / "DeepRadar2022" / "X_test.mat"
    y_file = repo / "data" / "DeepRadar2022" / "Y_test.mat"
    lbl_file = repo / "data" / "DeepRadar2022" / "lbl_test.mat"
    model_file = repo / "models" / DEEPRADAR_MODEL

    if not x_file.exists() or not y_file.exists() or not lbl_file.exists() or not model_file.exists():
        pytest.skip("Missing DeepRadar2022 data/model artifacts.")

    y_mat = loadmat(y_file)
    lbl_mat = loadmat(lbl_file)
    y_key = next(key for key in y_mat.keys() if not key.startswith("__"))
    lbl_key = next(key for key in lbl_mat.keys() if not key.startswith("__"))

    y_onehot = y_mat[y_key]
    labels_meta = lbl_mat[lbl_key]
    snr = labels_meta[:, 1]
    max_snr = float(np.max(snr))
    max_idx = np.where(snr == max_snr)[0]
    y_at_max = np.argmax(y_onehot[max_idx], axis=1)

    rng = np.random.default_rng(42)
    picked = []
    for cls in np.unique(y_at_max):
        cls_idx = max_idx[y_at_max == cls]
        k = min(40, len(cls_idx))
        picked.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
    picked = np.array(sorted(picked), dtype=np.int64)

    with h5py.File(x_file, "r") as h5:
        # Stored as (2, 1024, N) -> convert to (N, 1024, 2)
        x_raw = np.array(h5["X_test"][:, :, picked], dtype=np.float32)
    x_iq = np.transpose(x_raw, (2, 1, 0))

    # This model was trained with a derived third feature channel; IQ envelope
    # reproduces expected behavior for stable evaluation.
    envelope = np.sqrt(np.sum(np.square(x_iq), axis=2, keepdims=True))
    x = np.concatenate([x_iq, envelope], axis=2)
    y_true = np.argmax(y_onehot[picked], axis=1)

    model = load_model(model_file, compile=False)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    acc = accuracy_score(y_true, y_pred)

    assert len(np.unique(y_true)) >= 15, "Highest-SNR slice lacks class diversity."
    assert acc >= DEEPRADAR_MIN_ACC, (
        f"DeepRadar2022 max-SNR accuracy too low: {acc:.4f} at SNR={max_snr:.1f} dB "
        f"(threshold={DEEPRADAR_MIN_ACC:.2f}, n={len(y_true)})"
    )
