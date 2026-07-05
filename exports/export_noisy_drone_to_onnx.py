#!/usr/bin/env python3
"""Export the canonical NoisyDroneRFv2 Keras model to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/noisy_drone_vgg.yaml")
    parser.add_argument(
        "--keras-model",
        "--checkpoint",
        dest="keras_model",
        help="Override Keras checkpoint path.",
    )
    parser.add_argument(
        "--out",
        default="models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx",
        help="ONNX output path.",
    )
    parser.add_argument(
        "--sample-out",
        default="models/noisy_drone_rf_v2/sample_input.npy",
        help="Saved sample input for ONNX/TensorRT validation.",
    )
    parser.add_argument(
        "--labels-out",
        default="models/noisy_drone_rf_v2/labels.json",
        help="Saved class label list.",
    )
    parser.add_argument(
        "--sample-iq",
        type=Path,
        help="Optional NoisyDroneRFv2 .pt IQ file to convert into sample_input.npy.",
    )
    parser.add_argument("--sample-snr", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    from rf_signal_intelligence.config import load_yaml_config, resolve_path
    from rf_signal_intelligence.data.noisy_drone import (
        build_manifest,
        label_names_from_class_stats,
    )
    from rf_signal_intelligence.workflows.noisy_drone_vgg import (
        export_noisy_drone_vgg_to_onnx,
        prepare_spectrogram,
        spectrogram_config_from_mapping,
    )

    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    config.setdefault("project_root", str(config_path.resolve().parents[1]))
    config.setdefault("export", {})["onnx_path"] = args.out
    if args.keras_model:
        config.setdefault("model", {})["checkpoint"] = args.keras_model

    onnx_path = export_noisy_drone_vgg_to_onnx(config)

    project_root = resolve_path(config.get("project_root", "."))
    sample_out = resolve_path(args.sample_out, base_dir=project_root)
    labels_out = resolve_path(args.labels_out, base_dir=project_root)
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    spec_cfg = spectrogram_config_from_mapping(config)
    if args.sample_iq is not None:
        sample = prepare_spectrogram(
            args.sample_iq,
            snr=args.sample_snr,
            cache_dir=None,
            config=spec_cfg,
        )
    else:
        sample = np.zeros(spec_cfg.input_shape, dtype=np.float32)
    np.save(sample_out, sample.astype(np.float32)[None, ...])

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
    labels_out.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"Wrote ONNX: {onnx_path}")
    print(f"Wrote sample input: {sample_out}")
    print(f"Wrote labels: {labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
