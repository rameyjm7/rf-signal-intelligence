"""Reusable evaluation workflows for RML2016, RML2018, and DeepRadar2022."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rf_signal_intelligence.config import resolve_path
from rf_signal_intelligence.data.deepradar import load_deepradar_test_split
from rf_signal_intelligence.data.rml import (
    load_rml2016_pickle,
    load_rml2018_per_snr,
    load_rml2018_split,
    parse_classes_file,
    rml2016_arrays,
)
from rf_signal_intelligence.evaluation import classification_metrics, write_metrics_json
from rf_signal_intelligence.plots import (
    accuracy_by_snr,
    per_class_accuracy_by_snr,
    write_overall_snr_csv,
    write_per_class_snr_csv,
)


def _project_root(config: dict[str, Any]) -> Path:
    return resolve_path(config.get("project_root", "."))


def _outputs_dir(config: dict[str, Any], default: str) -> Path:
    output = resolve_path(config.get("outputs_dir", default), base_dir=_project_root(config))
    output.mkdir(parents=True, exist_ok=True)
    return output


def evaluate_rml2016(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the configured RML2016 Keras model and write metrics/SNR tables."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import load_model

    root = _project_root(config)
    data_path = resolve_path(config["dataset"]["pkl"], base_dir=root)
    model_path = resolve_path(config["model"]["checkpoint"], base_dir=root)
    outputs = _outputs_dir(config, "outputs/rml2016_eval")

    x_all, y_text, class_names = rml2016_arrays(load_rml2016_pickle(data_path))
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_text)
    _, x_test, _, y_test = train_test_split(
        x_all,
        y_all,
        test_size=float(config.get("evaluation", {}).get("test_size", 0.2)),
        random_state=int(config.get("random_state", 42)),
    )

    model = load_model(model_path, compile=False)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    metrics = classification_metrics(y_test, y_pred)
    payload = {
        "model": config["model"].get("id", "rml2016_lstm_rnn_2024"),
        "model_path": str(model_path),
        "classes": list(encoder.classes_),
        "test_samples": int(len(y_test)),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
    }
    write_metrics_json(outputs / "40_rml2016_eval_metrics.json", payload)

    snr_test = x_test[:, 0, 2].astype(np.int64)
    snr_values, overall = accuracy_by_snr(y_test, y_pred, snr_test)
    _, grid = per_class_accuracy_by_snr(y_test, y_pred, snr_test, n_classes=len(class_names))
    write_overall_snr_csv(outputs / "40_rml2016_accuracy_vs_snr.csv", snr_values, overall)
    write_per_class_snr_csv(
        outputs / "40_rml2016_accuracy_vs_snr_per_modulation.csv",
        list(encoder.classes_),
        snr_values,
        grid,
    )
    return payload


def evaluate_rml2018(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the configured RML2018 Keras model and write metrics/SNR tables."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import load_model

    root = _project_root(config)
    dataset = config["dataset"]
    eval_cfg = config.get("evaluation", {})
    h5_path = resolve_path(dataset["hdf5"], base_dir=root)
    classes_path = resolve_path(dataset["classes"], base_dir=root)
    model_path = resolve_path(config["model"]["checkpoint"], base_dir=root)
    outputs = _outputs_dir(config, "outputs/rml2018_eval")

    x_all, y_labels, _ = load_rml2018_split(
        h5_path,
        classes_path,
        snr_min_db=int(dataset.get("snr_min_db", -6)),
        snr_max_db=int(dataset.get("snr_max_db", 30)),
        max_per_class=int(dataset.get("max_samples_per_class", 3000)),
        random_state=int(config.get("random_state", 42)),
    )
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_labels)
    _, x_eval, _, y_eval = train_test_split(
        x_all,
        y_all,
        test_size=float(eval_cfg.get("test_size", 0.2)),
        stratify=y_all,
        random_state=int(config.get("random_state", 42)),
    )

    model = load_model(model_path, compile=False)
    y_pred = np.argmax(model.predict(x_eval, verbose=0), axis=1)
    metrics = classification_metrics(y_eval, y_pred)

    x_snr, y_snr_labels, snr_vals, _ = load_rml2018_per_snr(
        h5_path,
        classes_path,
        snr_min_db=int(dataset.get("snr_min_db", -6)),
        snr_max_db=int(dataset.get("snr_max_db", 30)),
        max_per_class_per_snr=int(eval_cfg.get("max_per_class_per_snr", 200)),
    )
    y_snr_true = encoder.transform(y_snr_labels)
    y_snr_pred = np.argmax(model.predict(x_snr, verbose=0), axis=1)
    snr_values, grid = per_class_accuracy_by_snr(
        y_snr_true,
        y_snr_pred,
        snr_vals,
        n_classes=len(encoder.classes_),
    )
    write_per_class_snr_csv(
        outputs / "41_rml2018_accuracy_per_snr_per_modulation.csv",
        list(encoder.classes_),
        snr_values,
        grid,
    )

    payload = {
        "model": config["model"].get("id", "rml2018_lstm_rnn"),
        "model_path": str(model_path),
        "classes": list(encoder.classes_),
        "test_samples": int(len(y_eval)),
        "per_snr_samples": int(len(y_snr_true)),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "per_snr_accuracy": float(accuracy_score(y_snr_true, y_snr_pred)),
    }
    write_metrics_json(outputs / "41_rml2018_eval_metrics.json", payload)
    return payload


def evaluate_deepradar2022(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the configured DeepRadar2022 Keras model and write metrics/SNR tables."""
    from tensorflow.keras.models import load_model

    root = _project_root(config)
    dataset = config["dataset"]
    x_path = resolve_path(dataset["x_test"], base_dir=root)
    y_path = resolve_path(dataset["y_test"], base_dir=root)
    lbl_path = resolve_path(dataset["lbl_test"], base_dir=root)
    model_path = resolve_path(config["model"]["checkpoint"], base_dir=root)
    outputs = _outputs_dir(config, "outputs/deepradar2022_eval")

    x_eval, y_true, snr_test = load_deepradar_test_split(x_path, y_path, lbl_path)
    model = load_model(model_path, compile=False)
    y_pred = np.argmax(model.predict(x_eval, verbose=0), axis=1)
    metrics = classification_metrics(y_true, y_pred)

    payload = {
        "model": config["model"].get("id", "deepradar2022_cnn_bilstm_final"),
        "model_path": str(model_path),
        "test_samples": int(len(y_true)),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
    }
    if snr_test is not None:
        snr_values, overall = accuracy_by_snr(y_true, y_pred, snr_test)
        _, grid = per_class_accuracy_by_snr(
            y_true,
            y_pred,
            snr_test,
            n_classes=int(max(y_true.max(), y_pred.max()) + 1),
        )
        class_names = [f"class_{idx}" for idx in range(grid.shape[0])]
        write_overall_snr_csv(outputs / "42_deepradar2022_accuracy_vs_snr_line.csv", snr_values, overall)
        write_per_class_snr_csv(
            outputs / "42_deepradar2022_accuracy_vs_snr_per_class.csv",
            class_names,
            snr_values,
            grid,
        )
    write_metrics_json(outputs / "42_deepradar2022_eval_metrics.json", payload)
    return payload


def rml2018_class_names(classes_path: str | Path) -> list[str]:
    """Expose class parsing to pipelines without importing dataset libraries."""
    return parse_classes_file(classes_path)
