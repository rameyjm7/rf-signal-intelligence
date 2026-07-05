"""Reusable loaders for RML2016 and RML2018 datasets."""

from __future__ import annotations

import ast
import pickle
import random
import re
from pathlib import Path

import numpy as np


def append_snr_channel(iq: np.ndarray, snr: float) -> np.ndarray:
    """Append a constant SNR feature channel to an IQ tensor."""
    signal = np.asarray(iq, dtype=np.float32)
    snr_col = np.full((signal.shape[0], 1), snr, dtype=np.float32)
    return np.hstack([signal, snr_col]).astype(np.float32)


def parse_classes_file(path: str | Path) -> list[str]:
    """Parse `classes = [...]` style class files used by RML2018."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"classes\s*=\s*(\[[\s\S]*?\])", text)
    if match:
        return [str(item) for item in ast.literal_eval(match.group(1))]
    return [str(item) for item in ast.literal_eval(text.split("=")[-1].strip())]


def load_rml2016_pickle(path: str | Path) -> dict:
    """Load the RML2016.10a pickle dictionary."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def rml2016_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert an RML2016 dictionary into `(samples, labels, class_names)`."""
    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    for (modulation, snr), signals in data.items():
        for signal in signals:
            iq = np.vstack([signal[0], signal[1]]).T.astype(np.float32)
            x_rows.append(append_snr_channel(iq, snr))
            y_rows.append(str(modulation))
    classes = sorted(set(y_rows))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows), classes


def sample_rml2016_high_snr(
    data: dict,
    *,
    n_per_class: int = 200,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample the highest-SNR RML2016 slice for cross-dataset diagnostics."""
    rng = np.random.default_rng(random_state)
    classes = sorted({mod for (mod, _snr) in data})
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    max_snr = max(snr for (_mod, snr) in data)
    rows: list[np.ndarray] = []
    y_idx: list[int] = []

    for modulation in classes:
        signals = data[(modulation, max_snr)]
        take = min(n_per_class, len(signals))
        picks = rng.choice(len(signals), size=take, replace=False)
        for idx in picks:
            signal = signals[int(idx)]
            iq = np.vstack([signal[0], signal[1]]).T.astype(np.float32)
            rows.append(append_snr_channel(iq, max_snr))
            y_idx.append(class_to_idx[modulation])
    return np.asarray(rows, dtype=np.float32), np.asarray(y_idx, dtype=np.int64), classes


def load_rml2018_split(
    h5_path: str | Path,
    classes_path: str | Path,
    *,
    snr_min_db: int = -6,
    snr_max_db: int = 30,
    max_per_class: int | None = 3000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a class-balanced RML2018 split aligned with notebook 31/41 preprocessing."""
    import h5py

    class_list = parse_classes_file(classes_path)
    rng = random.Random(random_state)
    with h5py.File(h5_path, "r") as handle:
        x_all = handle["X"][:]
        y_all = handle["Y"][:]
        z_all = handle["Z"][:]

    per_class: dict[str, list[np.ndarray]] = {label: [] for label in class_list}
    for idx in range(len(x_all)):
        snr = int(z_all[idx][0])
        if (snr > snr_min_db) and (snr <= snr_max_db):
            label = class_list[int(y_all[idx].argmax())]
            per_class[label].append(append_snr_channel(x_all[idx], snr))

    rows: list[np.ndarray] = []
    labels: list[str] = []
    for label, samples in per_class.items():
        rng.shuffle(samples)
        selected = samples[:max_per_class] if max_per_class else samples
        rows.extend(selected)
        labels.extend([label] * len(selected))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), class_list


def load_rml2018_per_snr(
    h5_path: str | Path,
    classes_path: str | Path,
    *,
    snr_min_db: int = -6,
    snr_max_db: int = 30,
    max_per_class_per_snr: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load a capped per-class/per-SNR RML2018 evaluation slice."""
    import h5py

    class_list = parse_classes_file(classes_path)
    buckets: dict[tuple[str, int], list[np.ndarray]] = {}
    with h5py.File(h5_path, "r") as handle:
        x_all = handle["X"][:]
        y_all = handle["Y"][:]
        z_all = handle["Z"][:]

    for idx in range(len(x_all)):
        snr = int(z_all[idx][0])
        if (snr > snr_min_db) and (snr <= snr_max_db):
            label = class_list[int(y_all[idx].argmax())]
            key = (label, snr)
            bucket = buckets.setdefault(key, [])
            if len(bucket) < max_per_class_per_snr:
                bucket.append(append_snr_channel(x_all[idx], snr))

    rows: list[np.ndarray] = []
    labels: list[str] = []
    snrs: list[int] = []
    for (label, snr), samples in buckets.items():
        rows.extend(samples)
        labels.extend([label] * len(samples))
        snrs.extend([snr] * len(samples))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels),
        np.asarray(snrs, dtype=np.int64),
        class_list,
    )
