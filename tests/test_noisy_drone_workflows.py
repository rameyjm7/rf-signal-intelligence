from pathlib import Path

import numpy as np

from rf_signal_intelligence.cli import build_parser
from rf_signal_intelligence.data.noisy_drone import (
    build_manifest,
    coerce_iq_array,
    parse_noisy_drone_filename,
)
from rf_signal_intelligence.features.spectrogram import (
    SpectrogramConfig,
    find_burst_start,
    iq_to_full_complex_spectrogram,
)
from rf_signal_intelligence.noisy_drone_framing import (
    NoisyDroneFrameConfig,
    build_model_windows,
    frequency_shift_iq,
)
from rf_signal_intelligence.plots import modulation_accuracy_traces_by_snr
from rf_signal_intelligence.workflows.comparison import comparison_row
from rf_signal_intelligence.workflows.noisy_drone_vgg import (
    balanced_replay_records,
    split_noisy_drone_records,
)


def test_parse_noisy_drone_filename_extracts_metadata():
    record = parse_noisy_drone_filename("IQdata_sample123_target4_snr-14.pt")

    assert record.sample_id == 123
    assert record.target_raw == 4
    assert record.snr == -14


def test_build_manifest_filters_snr_and_assigns_label_indices(tmp_path: Path):
    for name in [
        "IQdata_sample1_target0_snr-10.pt",
        "IQdata_sample2_target0_snr20.pt",
        "IQdata_sample3_target4_snr24.pt",
    ]:
        (tmp_path / name).touch()

    records = build_manifest(tmp_path, min_snr_db=0)

    assert [record.sample_id for record in records] == [2, 3]
    assert [record.label_idx for record in records] == [0, 1]


def test_coerce_iq_array_handles_complex_and_channel_first_inputs():
    complex_iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    channel_first = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)

    assert coerce_iq_array(complex_iq).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert coerce_iq_array(channel_first).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_spectrogram_shape_is_stable_for_small_config():
    rng = np.random.default_rng(7)
    iq = rng.normal(size=(128, 2)).astype(np.float32)
    config = SpectrogramConfig(sample_len=128, nfft=32, hop=16, time_bins=8)

    spec = iq_to_full_complex_spectrogram(iq, config)

    assert spec.shape == (32, 8, 2)
    assert spec.dtype == np.float32
    assert np.isfinite(spec).all()


def test_find_burst_start_centers_high_power_region():
    iq = np.zeros((100, 2), dtype=np.float32)
    iq[70:80] = 10.0

    start = find_burst_start(iq, 20, smooth_samples=1)

    assert 60 <= start <= 70


def test_noisy_drone_framer_keeps_default_offset_metadata():
    rng = np.random.default_rng(11)
    iq = rng.normal(size=(128, 2)).astype(np.float32)
    config = NoisyDroneFrameConfig(
        window_samples=128,
        nfft=32,
        hop=16,
        time_bins=8,
        scan_windows=False,
    )

    windows, candidates, raw_iq = build_model_windows(iq, config)

    assert windows.shape == (1, 32, 8, 2)
    assert candidates == [{"start": 0, "frequency_offset_hz": 0.0}]
    assert raw_iq.shape == (128, 2)


def test_noisy_drone_framer_expands_frequency_offset_candidates():
    rng = np.random.default_rng(12)
    iq = rng.normal(size=(256, 2)).astype(np.float32)
    config = NoisyDroneFrameConfig(
        window_samples=128,
        nfft=32,
        hop=16,
        time_bins=8,
        scan_stride_samples=64,
        sample_rate_hz=20_000_000,
        frequency_offsets_hz=(-1_000_000.0, 0.0, 1_000_000.0),
    )

    windows, candidates, _ = build_model_windows(iq, config)
    starts = sorted({candidate["start"] for candidate in candidates})

    assert windows.shape[0] == len(starts) * 3
    assert {candidate["frequency_offset_hz"] for candidate in candidates} == {
        -1_000_000.0,
        0.0,
        1_000_000.0,
    }


def test_frequency_shift_iq_preserves_shape_and_changes_nonzero_offsets():
    iq = np.column_stack(
        (
            np.ones(64, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
        )
    )

    shifted = frequency_shift_iq(iq, 1_000_000.0, 20_000_000.0)

    assert shifted.shape == iq.shape
    assert shifted.dtype == np.float32
    assert not np.allclose(shifted, iq)


def test_comparison_row_reports_model_metadata(tmp_path: Path):
    model = tmp_path / "model.keras"
    model.write_bytes(b"1234")

    row = comparison_row(
        dataset="demo",
        model_name="model",
        model_family="test",
        source="metrics.json",
        accuracy=0.5,
        model_path=model,
    )

    assert row["model_exists"] is True
    assert row["model_size_mb"] == 0.0
    assert row["eval_accuracy"] == 0.5


def test_rfsi_parser_accepts_requested_subcommands():
    parser = build_parser()

    assert parser.parse_args(["evaluate", "--config", "configs/noisy_drone_vgg.yaml"]).command == "evaluate"
    assert parser.parse_args(["compare", "--config", "configs/evaluation_comparison.yaml"]).command == "compare"
    assert parser.parse_args(["export-onnx", "--config", "configs/noisy_drone_vgg.yaml"]).command == "export-onnx"
    assert parser.parse_args(["train", "--config", "configs/noisy_drone_vgg.yaml"]).command == "train"


def test_rfsi_export_onnx_defaults_to_validation_and_sidecars():
    parser = build_parser()

    args = parser.parse_args(
        [
            "export-onnx",
            "--config",
            "configs/noisy_drone_vgg.yaml",
            "--sample-out",
            "sample.npy",
            "--labels-out",
            "labels.json",
        ]
    )

    assert args.validate is True
    assert args.sample_out == "sample.npy"
    assert args.labels_out == "labels.json"


def test_rfsi_export_onnx_can_skip_validation():
    parser = build_parser()

    args = parser.parse_args(["export-onnx", "--config", "configs/noisy_drone_vgg.yaml", "--no-validate"])

    assert args.validate is False


def test_noisy_drone_train_split_and_replay_balance(tmp_path: Path):
    for target in [0, 1, 2]:
        for sample in range(10):
            (tmp_path / f"IQdata_sample{target * 100 + sample}_target{target}_snr20.pt").touch()
    records = build_manifest(tmp_path, min_snr_db=0)

    train_records, val_records, test_records = split_noisy_drone_records(
        records,
        test_size=0.20,
        validation_size=0.25,
        random_state=7,
    )
    replay_records = balanced_replay_records(train_records, samples_per_class=8, random_state=7)

    assert len(train_records) == 18
    assert len(val_records) == 6
    assert len(test_records) == 6
    assert {record.label_idx for record in train_records} == {0, 1, 2}
    assert len(replay_records) == 24
    assert {label: sum(record.label_idx == label for record in replay_records) for label in [0, 1, 2]} == {
        0: 8,
        1: 8,
        2: 8,
    }


def test_modulation_accuracy_traces_by_snr_sorts_by_peak_accuracy():
    class FakeModel:
        def predict(self, x, verbose=False):
            # Class 0 is always correct; class 1 is always predicted as class 0.
            pred = np.zeros((len(x), 2), dtype=np.float32)
            pred[:, 0] = 1.0
            return pred

    x_test = np.array(
        [
            [[0, 0, 0]],
            [[0, 0, 10]],
            [[0, 0, 0]],
            [[0, 0, 10]],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0, 0, 1, 1], dtype=np.int64)

    traces, snrs = modulation_accuracy_traces_by_snr(FakeModel(), x_test, y_test, ["a", "b"])

    assert snrs == [np.float32(0.0), np.float32(10.0)]
    assert traces[0][0] == "a"
    assert traces[0][2] == 100.0
    assert traces[1][0] == "b"
    assert traces[1][2] == 0.0
