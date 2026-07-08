"""Plotting and tabular artifact helpers shared by pipelines."""

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


def modulation_accuracy_traces_by_snr(model, x_test, y_test, class_names: list[str]):
    """Compute per-modulation SNR accuracy traces sorted by peak accuracy."""
    snr_values = sorted(set(x_test[:, 0, -1]))
    traces = []
    for class_index, class_name in enumerate(class_names):
        accuracies = []
        for snr in snr_values:
            mask = np.where((y_test == class_index) & (x_test[:, 0, -1] == snr))
            x_slice = x_test[mask]
            y_slice = y_test[mask]
            if len(y_slice) > 0:
                y_pred = np.argmax(model.predict(x_slice, verbose=False), axis=1)
                accuracy = float(np.mean(y_pred == y_slice)) * 100.0
            else:
                accuracy = np.nan
            accuracies.append(accuracy)

        valid = [value for value in accuracies if not np.isnan(value)]
        peak_accuracy = max(valid) if valid else 0.0
        peak_snr = snr_values[accuracies.index(peak_accuracy)] if peak_accuracy > 0 else None
        traces.append((class_name, accuracies, peak_accuracy, peak_snr))
    return sorted(traces, key=lambda row: row[2], reverse=True), snr_values


def plot_modulation_accuracy_v_snr(
    model,
    x_test,
    y_test,
    label_encoder,
    *,
    top_n: int = 6,
    bottom_n: int = 5,
):
    """Plot all, top, and bottom modulation accuracy-vs-SNR traces."""
    import matplotlib.pyplot as plt

    traces, snr_values = modulation_accuracy_traces_by_snr(
        model,
        x_test,
        y_test,
        list(label_encoder.classes_),
    )

    def plot_group(group, title):
        plt.figure(figsize=(12, 8))
        for modulation, accuracies, peak_accuracy, peak_snr in group:
            label = (
                f"{modulation} (Peak: {peak_accuracy:.2f}% at {peak_snr} dB)"
                if peak_accuracy > 0
                else modulation
            )
            plt.plot(snr_values, accuracies, "-o", label=label)
            if peak_accuracy > 0 and peak_snr is not None:
                plt.plot(peak_snr, peak_accuracy, "ro")
                plt.text(
                    peak_snr,
                    peak_accuracy + 1,
                    f"{peak_accuracy:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.3"},
                )
        plt.xlabel("SNR (dB)")
        plt.ylabel("Recognition Accuracy (%)")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.ylim(0, 110)
        plt.xlim(min(snr_values), max(snr_values))
        plt.show()

    plot_group(traces, "Recognition Accuracy vs. SNR per Modulation Type (All Classifications)")
    plot_group(
        traces[:top_n],
        f"Recognition Accuracy vs. SNR per Modulation Type (Top {top_n} by Peak Accuracy)",
    )
    plot_group(
        traces[-bottom_n:],
        f"Recognition Accuracy vs. SNR per Modulation Type (Bottom {bottom_n} by Peak Accuracy)",
    )
    return traces
