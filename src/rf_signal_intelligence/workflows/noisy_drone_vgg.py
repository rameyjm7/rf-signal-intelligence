"""NoisyDroneRFv2 VGG spectrogram workflows for CLI and notebooks."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from rf_signal_intelligence.config import resolve_path
from rf_signal_intelligence.data.noisy_drone import (
    NoisyDroneRecord,
    build_manifest,
    label_names_from_class_stats,
    load_pt_iq,
)
from rf_signal_intelligence.evaluation import classification_metrics, write_metrics_json
from rf_signal_intelligence.features.spectrogram import (
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


def prepare_spectrogram_windows(
    filepath: str | Path,
    *,
    snr: float,
    cache_dir: str | Path | None,
    config: SpectrogramConfig,
    n_windows: int = 1,
) -> np.ndarray:
    """Prepare one or more spectrogram windows around the strongest burst."""
    if n_windows <= 1:
        return prepare_spectrogram(filepath, snr=snr, cache_dir=cache_dir, config=config)[None, ...]

    iq = load_pt_iq(filepath)
    if iq.shape[0] < config.sample_len:
        iq = np.pad(iq, ((0, config.sample_len - iq.shape[0]), (0, 0)), mode="constant")
    center_start = find_burst_start(iq, config.sample_len, smooth_samples=config.burst_smooth_samples)
    if iq.shape[0] <= config.sample_len:
        starts = [0]
    else:
        stride = max(1, config.sample_len // max(4, n_windows + 1))
        offsets = (np.arange(n_windows) - ((n_windows - 1) / 2.0)) * stride
        starts = [
            int(np.clip(center_start + offset, 0, iq.shape[0] - config.sample_len))
            for offset in offsets
        ]
        starts = sorted(set(starts))
    return np.stack(
        [
            iq_to_full_complex_spectrogram(iq[start : start + config.sample_len, :2], config)
            for start in starts
        ],
        axis=0,
    ).astype(np.float32)


def split_noisy_drone_records(
    records: list[NoisyDroneRecord],
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 1961,
) -> tuple[list[NoisyDroneRecord], list[NoisyDroneRecord], list[NoisyDroneRecord]]:
    """Create the notebook-style train/validation/test split."""
    from sklearn.model_selection import train_test_split

    labels = np.array([int(record.label_idx) for record in records], dtype=np.int64)
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels[train_idx],
    )
    return (
        [records[int(idx)] for idx in train_idx],
        [records[int(idx)] for idx in val_idx],
        [records[int(idx)] for idx in test_idx],
    )


def balanced_replay_records(
    records: list[NoisyDroneRecord],
    *,
    samples_per_class: int = 0,
    random_state: int = 1961,
) -> list[NoisyDroneRecord]:
    """Balance a training split by replaying minority classes with replacement."""
    groups: dict[int, list[NoisyDroneRecord]] = {}
    for record in records:
        groups.setdefault(int(record.label_idx), []).append(record)
    target_n = int(samples_per_class) if samples_per_class > 0 else max(len(group) for group in groups.values())
    rng = np.random.default_rng(random_state)
    replay: list[NoisyDroneRecord] = []
    for label_idx in sorted(groups):
        group = groups[label_idx]
        choices = rng.choice(len(group), size=target_n, replace=len(group) < target_n)
        replay.extend(group[int(choice)] for choice in choices)
    order = rng.permutation(len(replay))
    return [replay[int(idx)] for idx in order]


def _record_counts(records: list[NoisyDroneRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(int(record.label_idx))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _write_replay_manifest(path: Path, records: list[NoisyDroneRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "sample_id", "target_raw", "label_idx", "snr"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "filepath": str(record.filepath),
                    "sample_id": int(record.sample_id),
                    "target_raw": int(record.target_raw),
                    "label_idx": int(record.label_idx),
                    "snr": int(record.snr),
                }
            )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def build_vgg_spectrogram_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    learning_rate: float = 1.5e-4,
    weight_decay: float = 1e-4,
):
    """Build the notebook-33 VGG-style full-complex spectrogram model."""
    import tensorflow as tf
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
        SpatialDropout2D,
    )
    from tensorflow.keras.models import Model

    def vgg_block(x, filters: int, blocks: int, block_name: str, dropout_rate: float):
        regularizer = tf.keras.regularizers.l2(weight_decay)
        for i in range(blocks):
            x = Conv2D(
                filters,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=regularizer,
                name=f"{block_name}_conv{i + 1}",
            )(x)
            x = BatchNormalization(name=f"{block_name}_bn{i + 1}")(x)
        x = MaxPooling2D((2, 2), name=f"{block_name}_pool")(x)
        return SpatialDropout2D(dropout_rate, name=f"{block_name}_spatial_dropout")(x)

    inputs = Input(shape=input_shape)
    x = vgg_block(inputs, 32, 2, "block1", 0.08)
    x = vgg_block(x, 64, 2, "block2", 0.12)
    x = vgg_block(x, 128, 3, "block3", 0.18)
    x = vgg_block(x, 192, 3, "block4", 0.24)
    x = Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        name="block5_conv1",
    )(x)
    x = BatchNormalization(name="block5_bn1")(x)
    x = SpatialDropout2D(0.30, name="block5_spatial_dropout")(x)
    x = GlobalAveragePooling2D(name="vgg_spectrogram_embedding")(x)
    x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="fc1")(x)
    x = Dropout(0.55, name="fc1_dropout")(x)
    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name="fc2")(x)
    x = Dropout(0.45, name="fc2_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    model = Model(inputs, outputs, name="noisy_drone_rf_v2_vgg_full_complex_spectrogram")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_noisy_drone_vgg(config: dict[str, Any]) -> dict[str, Any]:
    """Train or continue the configured NoisyDroneRFv2 VGG spectrogram model."""
    project_root = resolve_path(config.get("project_root", "."))
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    data_dir = resolve_path(dataset_cfg["data_dir"], base_dir=project_root)
    best_path = resolve_path(model_cfg["checkpoint"], base_dir=project_root)
    final_path = resolve_path(
        model_cfg.get("final_checkpoint", str(best_path).replace("_best.keras", "_final.keras")),
        base_dir=project_root,
    )
    outputs_dir = resolve_path(config.get("outputs_dir", "outputs/noisy_drone_rf_v2_eval"), base_dir=project_root)
    cache_dir = resolve_path(config.get("cache_dir", "cache/noisy_drone_rf_v2"), base_dir=project_root)
    spec_cfg = spectrogram_config_from_mapping(config)

    random_state = int(config.get("random_state", 1961))
    batch_size = int(train_cfg.get("batch_size", eval_cfg.get("batch_size", 8)))
    epochs = int(train_cfg.get("epochs", 50))
    shuffle_buffer = int(train_cfg.get("shuffle_buffer", 256))
    replay = _as_bool(train_cfg.get("replay_buffer"), default=True)
    replay_samples_per_class = int(train_cfg.get("replay_samples_per_class", 0))

    records = build_manifest(
        data_dir,
        min_snr_db=float(dataset_cfg.get("min_snr_db", -6)),
        data_fraction=float(dataset_cfg.get("data_fraction", 0.25)),
        random_state=random_state,
    )
    if not records:
        raise FileNotFoundError(f"No NoisyDroneRFv2 samples found under {data_dir}")

    train_records, val_records, _ = split_noisy_drone_records(
        records,
        test_size=float(eval_cfg.get("test_size", train_cfg.get("test_size", 0.20))),
        validation_size=float(train_cfg.get("validation_size", 0.20)),
        random_state=random_state,
    )
    fit_records = (
        balanced_replay_records(
            train_records,
            samples_per_class=replay_samples_per_class,
            random_state=random_state,
        )
        if replay
        else list(train_records)
    )

    model_dir = best_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    history_json_path = resolve_path(
        model_cfg.get("history_path", model_dir / "noisy_drone_rf_v2_vgg_full_complex_spectrogram_history.json"),
        base_dir=project_root,
    )
    history_csv_path = resolve_path(
        model_cfg.get("history_csv_path", model_dir / "noisy_drone_rf_v2_vgg_full_complex_spectrogram_training_history.csv"),
        base_dir=project_root,
    )
    manifest_path = resolve_path(
        train_cfg.get("manifest_path", outputs_dir / "33_noisy_drone_rf_v2_vgg_full_complex_replay_manifest.csv"),
        base_dir=project_root,
    )
    _write_replay_manifest(manifest_path, fit_records)

    import tensorflow as tf
    from sklearn.metrics import f1_score
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.models import load_model

    if _as_bool(train_cfg.get("mixed_precision"), default=True):
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")

    def make_generator(split_records: list[NoisyDroneRecord]):
        def gen():
            for record in split_records:
                yield (
                    prepare_spectrogram(record.filepath, snr=record.snr, cache_dir=cache_dir, config=spec_cfg),
                    np.int64(record.label_idx),
                    np.float32(1.0),
                )

        return gen

    def make_dataset(split_records: list[NoisyDroneRecord], *, shuffle: bool, repeat: bool):
        ds = tf.data.Dataset.from_generator(
            make_generator(split_records),
            output_signature=(
                tf.TensorSpec(shape=spec_cfg.input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        if shuffle:
            ds = ds.shuffle(
                min(len(split_records), shuffle_buffer),
                seed=random_state,
                reshuffle_each_iteration=True,
            )
        if repeat:
            ds = ds.repeat()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    targets = sorted({record.target_raw for record in records})
    label_names = label_names_from_class_stats(data_dir, targets)
    num_classes = len(targets)
    input_shape = spec_cfg.input_shape

    resume_path = None
    if _as_bool(train_cfg.get("resume"), default=True):
        for candidate in (final_path, best_path):
            if candidate.exists():
                resume_path = candidate
                break

    model = None
    if resume_path is not None:
        try:
            loaded_model = load_model(resume_path, compile=False)
            if tuple(loaded_model.input_shape[1:]) == tuple(input_shape):
                model = loaded_model
            else:
                resume_path = None
        except Exception:
            resume_path = None

    if model is None:
        model = build_vgg_spectrogram_model(
            input_shape,
            num_classes,
            learning_rate=float(train_cfg.get("learning_rate", 1.5e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=float(train_cfg.get("continue_learning_rate", 5e-5)),
                clipnorm=1.0,
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    class ValidationMacroF1(tf.keras.callbacks.Callback):
        def __init__(self, validation_records: list[NoisyDroneRecord], *, eval_windows: int, max_samples: int):
            super().__init__()
            self.validation_records = (
                validation_records[:max_samples] if max_samples > 0 else list(validation_records)
            )
            self.eval_windows = int(eval_windows)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            y_true = np.asarray([int(record.label_idx) for record in self.validation_records], dtype=np.int64)
            y_pred = []
            for record in self.validation_records:
                x = prepare_spectrogram_windows(
                    record.filepath,
                    snr=record.snr,
                    cache_dir=cache_dir,
                    config=spec_cfg,
                    n_windows=self.eval_windows,
                )
                probs = self.model.predict(x, batch_size=batch_size, verbose=0).mean(axis=0)
                y_pred.append(int(np.argmax(probs)))
            logs["val_macro_f1"] = float(
                f1_score(y_true, np.asarray(y_pred), average="macro", zero_division=0)
            )
            print(f" - val_macro_f1: {logs['val_macro_f1']:.4f}")

    train_ds = make_dataset(fit_records, shuffle=True, repeat=True)
    val_ds = make_dataset(val_records, shuffle=False, repeat=True)
    train_steps = int(math.ceil(len(fit_records) / batch_size))
    validation_steps = int(math.ceil(len(val_records) / batch_size))
    callbacks = [
        ValidationMacroF1(
            val_records,
            eval_windows=int(train_cfg.get("val_metric_windows", 1)),
            max_samples=int(train_cfg.get("val_metric_limit", 0)),
        ),
        ModelCheckpoint(best_path, monitor="val_macro_f1", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
            patience=int(train_cfg.get("reduce_lr_patience", 3)),
            min_lr=float(train_cfg.get("min_learning_rate", 1e-7)),
            verbose=1,
        ),
    ]

    existing_rows: list[dict[str, str]] = []
    if history_csv_path.exists():
        with history_csv_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    last_global_epoch = max((int(float(row.get("global_epoch", 0))) for row in existing_rows), default=0)
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=int(train_cfg.get("verbose", 1)),
    )
    model.save(final_path)

    n_epochs = len(next(iter(history.history.values()))) if history.history else 0
    run_rows: list[dict[str, Any]] = []
    for epoch_idx in range(n_epochs):
        row: dict[str, Any] = {
            "run_id": run_id,
            "epoch": epoch_idx + 1,
            "global_epoch": last_global_epoch + epoch_idx + 1,
            "resumed_from": str(resume_path) if resume_path is not None else "",
            "data_fraction": float(dataset_cfg.get("data_fraction", 0.25)),
            "sample_len": int(spec_cfg.sample_len),
            "batch_size": int(batch_size),
            "train_fit_samples": int(len(fit_records)),
        }
        for metric_name, metric_values in history.history.items():
            row[metric_name] = float(metric_values[epoch_idx])
        run_rows.append(row)

    combined_rows = [*existing_rows, *run_rows]
    if combined_rows:
        fieldnames = sorted({key for row in combined_rows for key in row})
        preferred = [
            "run_id",
            "epoch",
            "global_epoch",
            "resumed_from",
            "data_fraction",
            "sample_len",
            "batch_size",
            "train_fit_samples",
        ]
        fieldnames = [name for name in preferred if name in fieldnames] + [
            name for name in fieldnames if name not in preferred
        ]
        history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with history_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    metric_name = "val_macro_f1" if any("val_macro_f1" in row for row in combined_rows) else "val_accuracy"
    best_row = None
    if combined_rows and metric_name:
        best_row = max(
            (row for row in combined_rows if row.get(metric_name) not in {None, ""}),
            key=lambda row: float(row[metric_name]),
            default=None,
        )

    payload = {
        "model": model_cfg.get("id", "noisy_drone_rf_v2_vgg_full_complex_spectrogram"),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "history_csv_path": str(history_csv_path),
        "manifest_path": str(manifest_path),
        "run_id": run_id,
        "run_epochs": int(epochs),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "last_global_epoch": int(last_global_epoch + n_epochs),
        "best_global_epoch": int(float(best_row["global_epoch"])) if best_row else int(last_global_epoch + n_epochs),
        "best_metric_name": metric_name,
        "best_metric_value": float(best_row[metric_name]) if best_row else None,
        "history": history.history,
        "config": {
            "sample_len": int(spec_cfg.sample_len),
            "input_shape": list(input_shape),
            "num_classes": int(num_classes),
            "label_names": label_names,
            "data_fraction": float(dataset_cfg.get("data_fraction", 0.25)),
            "min_snr_db": float(dataset_cfg.get("min_snr_db", -6)),
            "nfft": int(spec_cfg.nfft),
            "hop": int(spec_cfg.hop),
            "time_bins": int(spec_cfg.time_bins),
            "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
            "representation": "full_capture_complex_stft_real_imag",
            "train_source_samples": int(len(train_records)),
            "validation_samples": int(len(val_records)),
            "train_fit_samples": int(len(fit_records)),
            "train_source_counts": _record_counts(train_records),
            "train_replay_counts": _record_counts(fit_records),
        },
    }
    history_json_path.parent.mkdir(parents=True, exist_ok=True)
    history_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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


def noisy_drone_export_artifact_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Resolve ONNX, sample input, and labels paths for deployment export."""
    project_root = resolve_path(config.get("project_root", "."))
    export_cfg = config.setdefault("export", {})
    onnx_path = resolve_path(
        export_cfg.get("onnx_path", "models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx"),
        base_dir=project_root,
    )
    sample_path = resolve_path(
        export_cfg.get("sample_input_path", "models/noisy_drone_rf_v2/sample_input.npy"),
        base_dir=project_root,
    )
    labels_path = resolve_path(
        export_cfg.get("labels_path", "models/noisy_drone_rf_v2/labels.json"),
        base_dir=project_root,
    )
    return onnx_path, sample_path, labels_path


def write_noisy_drone_export_bundle(
    config: dict[str, Any],
    *,
    sample_iq: str | Path | None = None,
    sample_snr: float = 30.0,
) -> dict[str, Any]:
    """Export ONNX and write deployment sidecars: sample input and labels."""
    project_root = resolve_path(config.get("project_root", "."))
    onnx_path = export_noisy_drone_vgg_to_onnx(config)
    _, sample_path, labels_path = noisy_drone_export_artifact_paths(config)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    spec_cfg = spectrogram_config_from_mapping(config)
    if sample_iq is not None:
        sample = prepare_spectrogram(
            sample_iq,
            snr=sample_snr,
            cache_dir=None,
            config=spec_cfg,
        )
    else:
        sample = np.zeros(spec_cfg.input_shape, dtype=np.float32)
    np.save(sample_path, sample.astype(np.float32)[None, ...])

    dataset_cfg = config.get("dataset", {})
    data_dir = resolve_path(dataset_cfg.get("data_dir", "."), base_dir=project_root)
    labels = None
    if data_dir.exists():
        records = build_manifest(
            data_dir,
            min_snr_db=float(dataset_cfg.get("min_snr_db", -6)),
            data_fraction=float(dataset_cfg.get("data_fraction", 1.0)),
            random_state=int(config.get("random_state", 1961)),
        )
        if records:
            labels = label_names_from_class_stats(data_dir, sorted({record.target_raw for record in records}))
    if labels is None:
        labels = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Noise", "Taranis", "Turnigy"]
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    inference_script_path = onnx_path.parent / "run_onnx_inference.sh"
    inference_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'PYTHON="${{PYTHON:-{project_root / ".venv" / "bin" / "python3"}}}"',
            'if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi',
            f'"$PYTHON" "{project_root / "exports" / "run_onnx_inference.py"}" \\',
            f'  --onnx "{onnx_path}" \\',
            f'  --input "{sample_path}" \\',
            f'  --labels "{labels_path}" \\',
            '  "$@"',
            "",
        ]
    )
    inference_script_path.write_text(inference_script, encoding="utf-8")
    inference_script_path.chmod(0o755)

    return {
        "onnx_path": str(onnx_path),
        "sample_input_path": str(sample_path),
        "labels_path": str(labels_path),
        "inference_script_path": str(inference_script_path),
        "input_shape": [None, *spec_cfg.input_shape],
        "labels": labels,
    }


def _timed_call(fn, iterations: int) -> tuple[np.ndarray, float]:
    result = fn()
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        result = fn()
    elapsed = (time.perf_counter() - start) / max(1, iterations)
    return np.asarray(result), elapsed


def validate_noisy_drone_onnx_export(
    config: dict[str, Any],
    *,
    iterations: int = 20,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """Validate ONNX Runtime inference against the configured Keras checkpoint."""
    project_root = resolve_path(config.get("project_root", "."))
    checkpoint = resolve_path(config["model"]["checkpoint"], base_dir=project_root)
    onnx_path, sample_path, labels_path = noisy_drone_export_artifact_paths(config)
    sample = np.load(sample_path).astype(np.float32)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    import onnx
    import onnxruntime as ort
    from tensorflow.keras.models import load_model

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    keras_model = load_model(checkpoint, compile=False)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    keras_probs, keras_latency = _timed_call(
        lambda: keras_model.predict(sample, verbose=0),
        iterations,
    )
    onnx_probs, onnx_latency = _timed_call(
        lambda: session.run(None, {input_name: sample})[0],
        iterations,
    )
    keras_probs = np.asarray(keras_probs[0], dtype=np.float64)
    onnx_probs = np.asarray(onnx_probs[0], dtype=np.float64)
    keras_idx = int(np.argmax(keras_probs))
    onnx_idx = int(np.argmax(onnx_probs))
    max_abs_error = float(np.max(np.abs(keras_probs - onnx_probs)))
    mean_abs_error = float(np.mean(np.abs(keras_probs - onnx_probs)))
    passed = bool(np.allclose(keras_probs, onnx_probs, rtol=rtol, atol=atol))
    return {
        "keras_top1": labels[keras_idx] if keras_idx < len(labels) else str(keras_idx),
        "keras_confidence": float(keras_probs[keras_idx]),
        "onnx_top1": labels[onnx_idx] if onnx_idx < len(labels) else str(onnx_idx),
        "onnx_confidence": float(onnx_probs[onnx_idx]),
        "top1_agreement": keras_idx == onnx_idx,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "keras_latency_ms": keras_latency * 1000.0,
        "onnx_latency_ms": onnx_latency * 1000.0,
        "onnx_providers": session.get_providers(),
        "passed_tolerance": passed,
        "rtol": float(rtol),
        "atol": float(atol),
    }
