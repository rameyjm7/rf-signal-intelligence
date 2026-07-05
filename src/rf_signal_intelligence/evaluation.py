"""Reusable evaluation helpers for notebook and CLI workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    """Common classification metrics used across notebooks."""

    accuracy: float
    macro_f1: float
    weighted_f1: float


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Compute common classification metrics with sklearn imported lazily."""
    from sklearn.metrics import accuracy_score, f1_score

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )


def write_metrics_json(path: str | Path, metrics: dict[str, Any] | ClassificationMetrics) -> Path:
    """Write metrics as pretty JSON and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics) if isinstance(metrics, ClassificationMetrics) else metrics
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def top_class(probabilities: np.ndarray, labels: list[str] | None = None) -> tuple[str | int, float]:
    """Return top class label/index and confidence from a probability vector."""
    probs = np.asarray(probabilities, dtype=np.float32)
    idx = int(np.argmax(probs))
    label: str | int = labels[idx] if labels is not None else idx
    return label, float(probs[idx])
