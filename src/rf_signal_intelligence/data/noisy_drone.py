"""NoisyDroneRFv2 dataset manifest and IQ loading utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NOISY_DRONE_FILENAME_RE = re.compile(
    r"IQdata_sample(?P<sample>\d+)_target(?P<target>-?\d+)_snr(?P<snr>-?\d+)\.pt$"
)


@dataclass(frozen=True)
class NoisyDroneRecord:
    """One NoisyDroneRFv2 IQ sample on disk."""

    filepath: Path
    sample_id: int
    target_raw: int
    snr: int
    label_idx: int | None = None


def parse_noisy_drone_filename(path: str | Path) -> NoisyDroneRecord:
    """Parse sample id, target, and SNR from a NoisyDroneRFv2 `.pt` filename."""
    filepath = Path(path)
    match = NOISY_DRONE_FILENAME_RE.search(filepath.name)
    if not match:
        raise ValueError(f"Not a NoisyDroneRFv2 IQ filename: {filepath.name}")
    return NoisyDroneRecord(
        filepath=filepath,
        sample_id=int(match.group("sample")),
        target_raw=int(match.group("target")),
        snr=int(match.group("snr")),
    )


def build_manifest(
    data_dir: str | Path,
    *,
    min_snr_db: float | None = None,
    data_fraction: float = 1.0,
    random_state: int = 1961,
) -> list[NoisyDroneRecord]:
    """Build a deterministic file manifest for NoisyDroneRFv2 samples.

    The optional class-balanced fraction mirrors the original notebooks while keeping
    this function dependency-light. It samples within each target class when
    `data_fraction < 1`.
    """
    data_path = Path(data_dir).expanduser()
    records = [
        parse_noisy_drone_filename(path)
        for path in sorted(data_path.rglob("IQdata_sample*_target*_snr*.pt"))
    ]
    if min_snr_db is not None:
        records = [record for record in records if record.snr >= min_snr_db]
    if not records:
        return []

    classes = {target: idx for idx, target in enumerate(sorted({r.target_raw for r in records}))}
    records = [
        NoisyDroneRecord(r.filepath, r.sample_id, r.target_raw, r.snr, classes[r.target_raw])
        for r in records
    ]

    if data_fraction >= 1.0:
        return sorted(records, key=lambda r: r.sample_id)
    if data_fraction <= 0.0:
        raise ValueError("data_fraction must be greater than 0")

    rng = np.random.default_rng(random_state)
    selected: list[NoisyDroneRecord] = []
    for target in sorted(classes):
        group = [record for record in records if record.target_raw == target]
        count = max(1, int(round(len(group) * data_fraction)))
        indices = np.sort(rng.choice(len(group), size=min(count, len(group)), replace=False))
        selected.extend(group[int(i)] for i in indices)
    return sorted(selected, key=lambda r: r.sample_id)


def coerce_iq_array(value: Any, *, source: str | Path = "<memory>") -> np.ndarray:
    """Coerce common saved IQ payload shapes into an `(n, 2)` float32 array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if np.iscomplexobj(arr):
            arr = np.stack([arr.real, arr.imag], axis=-1)
        else:
            if arr.size % 2 != 0:
                arr = arr[:-1]
            arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        elif arr.shape[-1] != 2:
            raise ValueError(f"Expected IQ final dimension 2, got {arr.shape} in {source}")
    else:
        if arr.shape[-1] == 2:
            arr = arr.reshape(-1, 2)
        elif arr.shape[0] == 2:
            arr = np.moveaxis(arr, 0, -1).reshape(-1, 2)
        else:
            raise ValueError(f"Expected IQ tensor with two channels, got {arr.shape} in {source}")
    return np.asarray(arr, dtype=np.float32)


def extract_iq_payload(value: Any, *, source: str | Path = "<memory>") -> Any:
    """Extract an IQ-like payload from common Torch save structures."""
    if isinstance(value, dict):
        preferred = (
            "x_iq",
            "iq",
            "IQ",
            "x",
            "X",
            "data",
            "samples",
            "signal",
            "waveform",
            "input",
            "features",
            "arr",
            "array",
        )
        for key in preferred:
            if key in value:
                return extract_iq_payload(value[key], source=source)
        for item in value.values():
            try:
                return extract_iq_payload(item, source=source)
            except (TypeError, ValueError, KeyError):
                continue
        raise KeyError(f"No IQ-like payload found in {source}; keys={list(value.keys())}")
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"Empty sequence payload in {source}")
        return extract_iq_payload(value[0], source=source)
    return value


def load_pt_iq(filepath: str | Path) -> np.ndarray:
    """Load a NoisyDroneRFv2 Torch `.pt` IQ file as an `(n, 2)` float32 array."""
    path = Path(filepath)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading NoisyDroneRFv2 .pt files requires torch. "
            'Install it with `pip install -e ".[noisy-drone]"` from the repo root.'
        ) from exc

    payload = torch.load(path, map_location="cpu")
    return coerce_iq_array(extract_iq_payload(payload, source=path), source=path)


def label_names_from_class_stats(data_dir: str | Path, target_values: list[int]) -> list[str]:
    """Read class names from `class_stats.csv` when available."""
    class_stats = Path(data_dir) / "class_stats.csv"
    if not class_stats.exists():
        return [f"target_{target}" for target in target_values]

    import csv

    lookup: dict[int, str] = {}
    with class_stats.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if "class_int" in row and "class" in row:
                lookup[int(row["class_int"])] = row["class"]
    return [lookup.get(target, f"target_{target}") for target in target_values]
