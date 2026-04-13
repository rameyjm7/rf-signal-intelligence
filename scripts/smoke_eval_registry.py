#!/usr/bin/env python3
"""Registry-driven smoke checks for artifacts and model inference."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dataset_artifact_paths(dataset: dict[str, Any], repo: Path) -> list[Path]:
    paths: list[Path] = []
    if "path" in dataset:
        paths.append(repo / dataset["path"])
    if "paths" in dataset:
        for rel_path in dataset["paths"].values():
            paths.append(repo / rel_path)
    if "labels_path" in dataset:
        paths.append(repo / dataset["labels_path"])
    return paths


def _load_rml2016_sample(data_path: Path) -> np.ndarray:
    with data_path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")

    (modulation, snr), signals = next(iter(data.items()))
    signal = signals[0]
    iq_signal = np.vstack([signal[0], signal[1]]).T
    snr_channel = np.full((iq_signal.shape[0], 1), snr, dtype=np.float32)
    sample = np.hstack([iq_signal.astype(np.float32), snr_channel])
    print(f"  dataset sample: {modulation} @ {snr} dB")
    return sample[None, ...]


def _load_rml2018_sample(data_path: Path) -> np.ndarray:
    import h5py

    with h5py.File(data_path, "r") as h5:
        x_iq = np.asarray(h5["X"][0], dtype=np.float32)
        snr = float(h5["Z"][0, 0])

    snr_channel = np.full((x_iq.shape[0], 1), snr, dtype=np.float32)
    sample = np.hstack([x_iq, snr_channel])
    print(f"  dataset sample: index=0 @ {snr:.1f} dB")
    return sample[None, ...]


def _load_deepradar_sample(paths: dict[str, str], repo: Path) -> np.ndarray:
    import h5py

    x_path = repo / paths["x_test"]
    with h5py.File(x_path, "r") as h5:
        x_raw = np.asarray(h5["X_test"][:, :, 0], dtype=np.float32)

    x_iq = np.transpose(x_raw, (1, 0))
    envelope = np.sqrt(np.sum(np.square(x_iq), axis=1, keepdims=True))
    sample = np.hstack([x_iq, envelope])
    print("  dataset sample: index=0")
    return sample[None, ...]


def _build_real_sample(dataset: dict[str, Any], repo: Path) -> np.ndarray:
    dataset_id = dataset["id"]
    if dataset_id == "rml2016":
        return _load_rml2016_sample(repo / dataset["path"])
    if dataset_id == "rml2018":
        return _load_rml2018_sample(repo / dataset["path"])
    if dataset_id == "deepradar2022":
        return _load_deepradar_sample(dataset["paths"], repo)
    raise ValueError(f"Unsupported dataset id: {dataset_id}")


def _dummy_input_from_model(model: Any) -> np.ndarray:
    shape = [dim if dim is not None else 1 for dim in model.input_shape]
    return np.zeros(shape, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registry-driven smoke checks.")
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Use one real sample from each dataset when available.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail if required model/data artifacts are missing.",
    )
    args = parser.parse_args()

    repo = _repo_root()
    dataset_registry = _load_yaml(repo / "configs" / "data_registry.yaml")
    model_registry = _load_yaml(repo / "configs" / "model_registry.yaml")

    datasets = {entry["id"]: entry for entry in dataset_registry["datasets"]}
    missing: list[str] = []

    print("Checking registry artifact availability...")
    for dataset_id, dataset in datasets.items():
        for path in _dataset_artifact_paths(dataset, repo):
            if not path.exists():
                missing.append(f"dataset:{dataset_id}:{path}")

    for model in model_registry["models"]:
        model_path = repo / model["path"]
        if not model_path.exists():
            missing.append(f"model:{model['id']}:{model_path}")

    if missing:
        print("Missing artifacts:")
        for item in missing:
            print(f"  - {item}")
        if args.require_artifacts:
            return 1
        print("Continuing because --require-artifacts was not set.")

    print("Running model smoke inference...")
    from tensorflow.keras.models import load_model

    for model_entry in model_registry["models"]:
        model_path = repo / model_entry["path"]
        dataset = datasets[model_entry["dataset"]]
        if not model_path.exists():
            print(f"- skip {model_entry['id']}: missing model file")
            continue

        print(f"- model {model_entry['id']}")
        try:
            model = load_model(model_path, compile=False)
        except Exception as exc:
            message = f"  failed to load {model_path}: {exc}"
            if args.require_artifacts:
                print(message)
                return 1
            print(f"{message}; skipping")
            continue

        if args.with_data:
            sample = _build_real_sample(dataset, repo)
        else:
            sample = _dummy_input_from_model(model)

        prediction = model.predict(sample, verbose=0)
        print(f"  input shape={sample.shape}, output shape={prediction.shape}")

    print("Smoke evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
