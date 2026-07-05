"""Plotting and tabular artifact helpers shared by notebooks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def accuracy_by_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute overall accuracy percentage for each SNR value."""
    snr_values = np.array(sorted(np.unique(snr)), dtype=np.int64)
    accuracy = []
    for value in snr_values:
        mask = snr == value
        accuracy.append(float(np.mean(y_pred[mask] == y_true[mask])) * 100.0)
    return snr_values, np.asarray(accuracy, dtype=np.float32)


def per_class_accuracy_by_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute class-by-SNR accuracy percentage grid."""
    snr_values = np.array(sorted(np.unique(snr)), dtype=np.int64)
    grid = np.full((n_classes, len(snr_values)), np.nan, dtype=np.float32)
    for class_idx in range(n_classes):
        class_mask = y_true == class_idx
        for snr_idx, value in enumerate(snr_values):
            mask = class_mask & (snr == value)
            if np.any(mask):
                grid[class_idx, snr_idx] = float(np.mean(y_pred[mask] == y_true[mask])) * 100.0
    return snr_values, grid


def write_overall_snr_csv(path: str | Path, snr_values: np.ndarray, accuracy: np.ndarray) -> Path:
    """Write overall SNR accuracy table."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr_db", "accuracy_percent"])
        for snr, acc in zip(snr_values, accuracy):
            writer.writerow([int(snr), f"{float(acc):.6f}"])
    return output


def write_per_class_snr_csv(
    path: str | Path,
    class_names: list[str],
    snr_values: np.ndarray,
    grid: np.ndarray,
) -> Path:
    """Write per-class SNR accuracy table."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class"] + [str(int(value)) for value in snr_values])
        for class_idx, name in enumerate(class_names):
            row = [
                "" if np.isnan(grid[class_idx, snr_idx]) else f"{float(grid[class_idx, snr_idx]):.6f}"
                for snr_idx in range(len(snr_values))
            ]
            writer.writerow([name, *row])
    return output
