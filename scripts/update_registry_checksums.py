#!/usr/bin/env python3
"""Populate SHA256 checksums for registry entries when files exist."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA_REG = REPO / "configs" / "data_registry.yaml"
MODEL_REG = REPO / "configs" / "model_registry.yaml"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _set_checksum(entry: dict[str, object], path: Path) -> None:
    entry.setdefault("checksum", {"algorithm": "sha256", "value": None})
    if path.exists():
        entry["checksum"]["value"] = _sha256(path)
    else:
        entry["checksum"]["value"] = None


def main() -> int:
    data_registry = _load_yaml(DATA_REG)
    for dataset in data_registry["datasets"]:
        if "path" in dataset:
            _set_checksum(dataset, REPO / dataset["path"])
        elif "paths" in dataset:
            first_key = next(iter(dataset["paths"]))
            _set_checksum(dataset, REPO / dataset["paths"][first_key])

    model_registry = _load_yaml(MODEL_REG)
    for model in model_registry["models"]:
        _set_checksum(model, REPO / model["path"])

    _write_yaml(DATA_REG, data_registry)
    _write_yaml(MODEL_REG, model_registry)
    print("Registry checksums updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
