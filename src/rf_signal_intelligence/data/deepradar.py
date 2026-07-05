"""Reusable DeepRadar2022 dataset helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def append_envelope_channel(x_iq: np.ndarray) -> np.ndarray:
    """Append IQ envelope magnitude as a third channel."""
    iq = np.asarray(x_iq, dtype=np.float32)
    envelope = np.sqrt(np.sum(np.square(iq), axis=2, keepdims=True))
    return np.concatenate([iq, envelope], axis=2).astype(np.float32)


def load_deepradar_test_split(
    x_path: str | Path,
    y_path: str | Path,
    lbl_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load DeepRadar2022 test split as `(X_eval, y_true, snr_test)`."""
    import h5py
    from scipy.io import loadmat

    with h5py.File(x_path, "r") as handle:
        x_raw = np.asarray(handle["X_test"][:], dtype=np.float32)
    x_iq = np.transpose(x_raw, (2, 1, 0))
    x_eval = append_envelope_channel(x_iq)

    y_mat = loadmat(y_path)
    lbl_mat = loadmat(lbl_path)
    y_key = next(key for key in y_mat if not key.startswith("__"))
    lbl_key = next(key for key in lbl_mat if not key.startswith("__"))
    y_onehot = y_mat[y_key]
    labels = lbl_mat[lbl_key]
    y_true = np.argmax(y_onehot, axis=1).astype(np.int64)
    snr_test = labels[:, 1].astype(np.int64) if labels.ndim == 2 and labels.shape[1] > 1 else None
    return x_eval, y_true, snr_test
