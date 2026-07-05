"""Cross-dataset comparison workflow extracted from notebook 50."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any | None:
    """Read JSON and return None when the artifact is missing."""
    artifact = Path(path)
    if not artifact.exists():
        return None
    return json.loads(artifact.read_text(encoding="utf-8"))


def model_size_mb(path: str | Path | None) -> float | None:
    """Return model size in MiB when the file exists."""
    if path is None:
        return None
    model_path = Path(path)
    if not model_path.exists():
        return None
    return round(model_path.stat().st_size / (1024 * 1024), 3)


def comparison_row(
    *,
    dataset: str,
    model_name: str,
    model_family: str,
    source: str | Path,
    split: str = "all_test",
    accuracy: float | None = None,
    macro_f1: float | None = None,
    weighted_f1: float | None = None,
    model_path: str | Path | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build one standardized cross-dataset comparison row."""
    model = Path(model_path) if model_path is not None else None
    return {
        "dataset": dataset,
        "model_name": model_name,
        "model_family": model_family,
        "split": split,
        "eval_accuracy": accuracy,
        "eval_macro_f1": macro_f1,
        "eval_weighted_f1": weighted_f1,
        "model_path": str(model) if model is not None else "",
        "model_exists": bool(model.exists()) if model is not None else None,
        "model_size_mb": model_size_mb(model),
        "source": str(source),
        "notes": notes,
    }


def build_comparison_rows(config: dict[str, Any], *, project_root: Path) -> list[dict[str, Any]]:
    """Build comparison rows from configured metric artifacts."""
    rows: list[dict[str, Any]] = []
    for item in config.get("metrics", []):
        source = project_root / item["source"]
        metrics = read_json(source)
        if metrics is None:
            if item.get("optional", True):
                continue
            raise FileNotFoundError(source)

        if isinstance(metrics, list):
            metric_rows = metrics
        elif isinstance(metrics, dict):
            metric_rows = [metrics]
        else:
            continue

        for metric in metric_rows:
            rows.append(
                comparison_row(
                    dataset=item["dataset"],
                    model_name=item.get("model_name") or metric.get("model") or metric.get("model_name", "unknown"),
                    model_family=item.get("model_family", "unknown"),
                    split=item.get("split") or metric.get("split", "all_test"),
                    accuracy=metric.get(item.get("accuracy_key", "accuracy")),
                    macro_f1=metric.get(item.get("macro_f1_key", "macro_f1")),
                    weighted_f1=metric.get(item.get("weighted_f1_key", "weighted_f1")),
                    model_path=project_root / item["model_path"] if item.get("model_path") else None,
                    source=source,
                    notes=item.get("notes", ""),
                )
            )
    return rows


def write_comparison_outputs(
    rows: list[dict[str, Any]],
    *,
    output_csv: str | Path,
    output_json: str | Path,
) -> tuple[Path, Path]:
    """Write comparison rows as CSV and JSON."""
    import pandas as pd

    if not rows:
        raise FileNotFoundError("No comparison metric rows were found.")
    df = pd.DataFrame(rows)
    for col in ["eval_accuracy", "eval_macro_f1", "eval_weighted_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["dataset", "split", "eval_accuracy"], ascending=[True, True, False])

    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path
