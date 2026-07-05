#!/usr/bin/env python3
"""Validate ONNX inference against the source Keras model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras-model",
        default="models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras",
    )
    parser.add_argument(
        "--onnx",
        default="models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx",
    )
    parser.add_argument("--sample", default="models/noisy_drone_rf_v2/sample_input.npy")
    parser.add_argument("--labels", default="models/noisy_drone_rf_v2/labels.json")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def timed_call(fn, iterations: int) -> tuple[np.ndarray, float]:
    result = fn()
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        result = fn()
    elapsed = (time.perf_counter() - start) / max(1, iterations)
    return np.asarray(result), elapsed


def main() -> int:
    args = parse_args()
    sample = np.load(args.sample).astype(np.float32)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    import onnx
    import onnxruntime as ort
    from tensorflow.keras.models import load_model

    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)

    keras_model = load_model(args.keras_model, compile=False)
    session = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    keras_probs, keras_latency = timed_call(
        lambda: keras_model.predict(sample, verbose=0),
        args.iterations,
    )
    onnx_probs, onnx_latency = timed_call(
        lambda: session.run(None, {input_name: sample})[0],
        args.iterations,
    )
    keras_probs = np.asarray(keras_probs[0], dtype=np.float64)
    onnx_probs = np.asarray(onnx_probs[0], dtype=np.float64)

    keras_idx = int(np.argmax(keras_probs))
    onnx_idx = int(np.argmax(onnx_probs))
    max_abs_error = float(np.max(np.abs(keras_probs - onnx_probs)))
    mean_abs_error = float(np.mean(np.abs(keras_probs - onnx_probs)))
    payload = {
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
    }
    print(json.dumps(payload, indent=2))
    if not np.allclose(keras_probs, onnx_probs, rtol=args.rtol, atol=args.atol):
        raise SystemExit(
            f"ONNX output differs from Keras beyond tolerances: "
            f"max_abs_error={max_abs_error:.6g}, mean_abs_error={mean_abs_error:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
