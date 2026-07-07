#!/usr/bin/env python3
"""Classify raw NoisyDroneRF IQ using the shared live framing component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _preflight_cuda_for_tensorrt() -> None:
    """Initialize CUDA early on Jetson before heavier RFML imports run."""
    try:
        from cuda import cudart
    except ImportError:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            return
    count_result = cudart.cudaGetDeviceCount()
    if int(count_result[0]) != 0 or int(count_result[1]) <= 0:
        return
    cudart.cudaSetDevice(0)
    cudart.cudaFree(0)


_preflight_cuda_for_tensorrt()

from rf_signal_intelligence.live_noisy_drone_rf_classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LABEL_NAMES,
    load_iq_file,
)
from rf_signal_intelligence.noisy_drone_framing import (  # noqa: E402
    NoisyDroneFrameClassifier,
    NoisyDroneFrameConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iq-file", type=Path, required=True, help="Raw IQ file: .pt, .npy, .npz, .bin, .c64.")
    parser.add_argument("--backend", choices=("keras", "tensorrt"), default="keras")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Keras model path.")
    parser.add_argument("--engine", type=Path, help="TensorRT engine path for --backend tensorrt.")
    parser.add_argument("--labels", type=Path, help="Optional JSON label list. Defaults to NoisyDroneRF labels.")
    parser.add_argument("--target-class", choices=LABEL_NAMES, help="Expected/target class for window scoring.")
    parser.add_argument("--window-samples", type=int, default=1_048_576)
    parser.add_argument("--scan-stride-samples", type=int, default=262_144)
    parser.add_argument("--no-scan-windows", action="store_true")
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=1024)
    parser.add_argument("--time-bins", type=int, default=1024)
    parser.add_argument(
        "--window-score-mode",
        choices=("auto", "target", "non-noise", "raw"),
        default="auto",
    )
    parser.add_argument(
        "--decision-mode",
        choices=("hybrid", "raw", "non-noise"),
        default="hybrid",
    )
    parser.add_argument("--non-noise-threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--format", choices=("json", "table"), default="table")
    return parser.parse_args()


def load_labels(path: Path | None) -> list[str]:
    if path is None:
        return list(LABEL_NAMES)
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"Label file must be a JSON string list: {path}")
    return labels


def make_predictor(args: argparse.Namespace):
    if args.backend == "keras":
        from tensorflow.keras.models import load_model

        model = load_model(args.model, compile=False)

        def predict_batch(batch: np.ndarray) -> np.ndarray:
            return model.predict(batch, verbose=0)

        return predict_batch

    if args.engine is None:
        raise ValueError("--engine is required with --backend tensorrt")
    from deploy.run_tensorrt_engine_inference import TensorRtRunner

    runner = TensorRtRunner(args.engine)

    def predict_batch(batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.shape[0] == 1:
            return runner.infer(batch)
        rows = [runner.infer(batch[idx : idx + 1])[0] for idx in range(batch.shape[0])]
        return np.stack(rows, axis=0).astype(np.float32)

    predict_batch.close = runner.close  # type: ignore[attr-defined]
    return predict_batch


def print_table(payload: dict) -> None:
    print("NoisyDroneRF Frame Classification")
    print(f"  prediction : {payload['prediction']} ({payload['confidence']:.3f})")
    print(f"  raw top-1  : {payload['raw_prediction']} ({payload['raw_confidence']:.3f})")
    print(f"  non-noise  : {payload['best_non_noise']} ({payload['best_non_noise_confidence']:.3f})")
    if payload.get("target_class"):
        print(
            f"  target     : {payload['target_class']} "
            f"conditional={payload.get('target_conditional_confidence', 0.0):.3f} "
            f"correct={payload.get('correct')}"
        )
    print(
        "  top        : "
        + ", ".join(f"{item['label']}={item['confidence']:.3f}" for item in payload["top"])
    )
    scan = payload["scan"]
    print(
        f"  window     : start={scan['selected_start']} "
        f"mode={scan['resolved_window_score_mode']} candidates={len(scan['candidates'])}"
    )
    stats = payload["capture_stats"]
    print(
        f"  capture    : power={stats['power_db']:.1f} dB peak={stats['peak']:.3f} "
        f"clip={stats['mag_fullscale_pct']:.2f}%"
    )


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    config = NoisyDroneFrameConfig(
        window_samples=args.window_samples,
        nfft=args.nfft,
        hop=args.hop,
        time_bins=args.time_bins,
        scan_windows=not args.no_scan_windows,
        scan_stride_samples=args.scan_stride_samples,
        window_score_mode=args.window_score_mode,
        decision_mode=args.decision_mode,
        non_noise_threshold=args.non_noise_threshold,
        top_k=args.top_k,
    )
    predictor = make_predictor(args)
    try:
        classifier = NoisyDroneFrameClassifier(predictor, labels=labels, config=config)
        payload = classifier.classify_iq(load_iq_file(args.iq_file), target_label=args.target_class)
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print_table(payload)
    finally:
        close = getattr(predictor, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
