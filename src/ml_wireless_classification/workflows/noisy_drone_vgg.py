"""NoisyDroneRFv2 VGG spectrogram workflows for CLI and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml_wireless_classification.config import resolve_path
from ml_wireless_classification.data.noisy_drone import (
    build_manifest,
    label_names_from_class_stats,
    load_pt_iq,
)
from ml_wireless_classification.evaluation import classification_metrics, write_metrics_json
from ml_wireless_classification.features.spectrogram import (
    SpectrogramConfig,
    find_burst_start,
    iq_to_full_complex_spectrogram,
    safe_load_spectrogram_cache,
    write_spectrogram_cache_atomic,
)


def spectrogram_config_from_mapping(config: dict[str, Any]) -> SpectrogramConfig:
    """Create a spectrogram config from a workflow config mapping."""
    feature = config.get("features", {})
    return SpectrogramConfig(
        sample_len=int(feature.get("sample_len", 1_048_576)),
        nfft=int(feature.get("nfft", 1024)),
        hop=int(feature.get("hop", 1024)),
        time_bins=int(feature.get("time_bins", 1024)),
        burst_smooth_samples=int(feature.get("burst_smooth_samples", 512)),
    )


def spectrogram_cache_path(
    filepath: str | Path,
    *,
    cache_dir: str | Path,
    config: SpectrogramConfig,
    snr: float,
) -> Path:
    """Return the cache path used for a prepared spectrogram."""
    src = Path(filepath)
    name = (
        f"{src.stem}_full_complex_len{config.sample_len}_nfft{config.nfft}"
        f"_hop{config.hop}_tb{config.time_bins}_snr{int(float(snr))}.npz"
    )
    return Path(cache_dir) / name


def prepare_spectrogram(
    filepath: str | Path,
    *,
    snr: float,
    cache_dir: str | Path | None,
    config: SpectrogramConfig,
) -> np.ndarray:
    """Load IQ and prepare one cached full-complex spectrogram."""
    cache_path = (
        spectrogram_cache_path(filepath, cache_dir=cache_dir, config=config, snr=snr)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        cached = safe_load_spectrogram_cache(cache_path, config.input_shape)
        if cached is not None:
            return cached

    iq = load_pt_iq(filepath)
    if iq.shape[0] < config.sample_len:
        iq = np.pad(iq, ((0, config.sample_len - iq.shape[0]), (0, 0)), mode="constant")
    start = find_burst_start(iq, config.sample_len, smooth_samples=config.burst_smooth_samples)
    spec = iq_to_full_complex_spectrogram(iq[start : start + config.sample_len, :2], config)
    if cache_path is not None:
        write_spectrogram_cache_atomic(cache_path, spec)
    return spec


def evaluate_noisy_drone_vgg(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a configured NoisyDroneRFv2 Keras model and write metrics."""
    project_root = resolve_path(config.get("project_root", "."))
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    eval_cfg = config.get("evaluation", {})

    data_dir = resolve_path(dataset_cfg["data_dir"], base_dir=project_root)
    checkpoint = resolve_path(model_cfg["checkpoint"], base_dir=project_root)
    outputs_dir = resolve_path(config.get("outputs_dir", "outputs/noisy_drone_rf_v2_eval"), base_dir=project_root)
    cache_dir = resolve_path(config.get("cache_dir", "cache/noisy_drone_rf_v2"), base_dir=project_root)
    spec_cfg = spectrogram_config_from_mapping(config)

    records = build_manifest(
        data_dir,
        min_snr_db=float(dataset_cfg.get("min_snr_db", -6)),
        data_fraction=float(dataset_cfg.get("data_fraction", 0.25)),
        random_state=int(config.get("random_state", 1961)),
    )
    if not records:
        raise FileNotFoundError(f"No NoisyDroneRFv2 samples found under {data_dir}")

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import load_model

    labels = np.array([record.label_idx for record in records], dtype=np.int64)
    indices = np.arange(len(records))
    _, test_idx = train_test_split(
        indices,
        test_size=float(eval_cfg.get("test_size", 0.20)),
        random_state=int(config.get("random_state", 1961)),
        stratify=labels,
    )
    if eval_cfg.get("limit"):
        test_idx = test_idx[: int(eval_cfg["limit"])]

    model = load_model(checkpoint, compile=False)
    probs = []
    y_true = []
    for idx in test_idx:
        record = records[int(idx)]
        x = prepare_spectrogram(record.filepath, snr=record.snr, cache_dir=cache_dir, config=spec_cfg)
        pred = model.predict(x[None, ...], batch_size=int(eval_cfg.get("batch_size", 8)), verbose=0)
        probs.append(pred[0])
        y_true.append(record.label_idx)

    probabilities = np.asarray(probs, dtype=np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred = probabilities.argmax(axis=1)
    metrics = classification_metrics(y_true_arr, y_pred)
    targets = sorted({record.target_raw for record in records})
    payload = {
        "model": model_cfg.get("id", "noisy_drone_rf_v2_vgg_full_complex_spectrogram"),
        "model_path": str(checkpoint),
        "label_names": label_names_from_class_stats(data_dir, targets),
        "test_samples": int(len(y_true_arr)),
        "input_shape": list(spec_cfg.input_shape),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
    }
    output_path = outputs_dir / config.get(
        "metrics_filename", "noisy_drone_rf_v2_vgg_full_complex_spectrogram_metrics.json"
    )
    write_metrics_json(output_path, payload)
    return payload


def export_noisy_drone_vgg_to_onnx(config: dict[str, Any]) -> Path:
    """Export the configured Keras model to ONNX using tf2onnx."""
    project_root = resolve_path(config.get("project_root", "."))
    checkpoint = resolve_path(config["model"]["checkpoint"], base_dir=project_root)
    onnx_path = resolve_path(config["export"]["onnx_path"], base_dir=project_root)
    spec_cfg = spectrogram_config_from_mapping(config)

    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(checkpoint, compile=False)
    signature = (
        tf.TensorSpec((None, *spec_cfg.input_shape), tf.float32, name=config["export"].get("input_name", "input")),
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(model, input_signature=signature, output_path=str(onnx_path))
    return onnx_path
