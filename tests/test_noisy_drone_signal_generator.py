from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from rf_signal_intelligence.noisy_drone_signal_generator import (
    build_source_cache,
    iq_summary,
    make_burst,
    parse_args,
    parse_class_list,
)


def test_parse_class_list_accepts_all_non_noise_classes():
    assert parse_class_list("all") == ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Taranis", "Turnigy"]


def test_parse_class_list_rejects_unknown_class():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_class_list("DJI,Nope")


def test_iq_file_source_cache_loads_once(tmp_path: Path):
    iq_path = tmp_path / "sample.npy"
    np.save(iq_path, np.ones((8, 2), dtype=np.float32))

    args = parse_args(["--source", "iq-file", "--iq-file", str(iq_path)])
    cache = build_source_cache(args)

    assert cache == {}


def test_make_burst_prepares_cached_iq_file(tmp_path: Path):
    iq_path = tmp_path / "sample.npy"
    iq = np.stack(
        [
            np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            np.linspace(1.0, -1.0, 8, dtype=np.float32),
        ],
        axis=-1,
    )
    np.save(iq_path, iq)
    args = parse_args(
        [
            "--source",
            "iq-file",
            "--iq-file",
            str(iq_path),
            "--amplitude",
            "0.1",
            "--pad-sec",
            "0",
        ]
    )
    cache = build_source_cache(args)

    source_desc, prepared = make_burst(args, class_name="iq-file", cycle_index=0, cache=cache)

    assert source_desc == str(iq_path)
    assert prepared.shape == (8, 2)
    assert np.isclose(np.max(np.abs(prepared[:, 0] + 1j * prepared[:, 1])), 0.1)
    assert "samples=8" in iq_summary(prepared)
