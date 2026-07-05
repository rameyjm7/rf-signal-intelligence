#!/usr/bin/env python3
"""Run Jetson-oriented ONNX/TensorRT-provider inference for the RF model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        default="models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx",
    )
    parser.add_argument("--sample", default="models/noisy_drone_rf_v2/sample_input.npy")
    parser.add_argument("--labels", default="models/noisy_drone_rf_v2/labels.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--providers",
        default="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers in priority order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample = np.load(args.sample).astype(np.float32)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    requested = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    providers = [provider for provider in requested if provider in available]
    if not providers:
        raise RuntimeError(f"None of the requested providers are available. Available: {sorted(available)}")

    session = ort.InferenceSession(args.onnx, providers=providers)
    input_name = session.get_inputs()[0].name

    probs = session.run(None, {input_name: sample})[0]
    start = time.perf_counter()
    for _ in range(max(1, args.iterations)):
        probs = session.run(None, {input_name: sample})[0]
    elapsed = time.perf_counter() - start

    probs = np.asarray(probs[0], dtype=np.float64)
    top_idx = int(np.argmax(probs))
    payload = {
        "prediction": labels[top_idx] if top_idx < len(labels) else str(top_idx),
        "confidence": float(probs[top_idx]),
        "providers": session.get_providers(),
        "iterations": int(args.iterations),
        "avg_latency_ms": elapsed / max(1, args.iterations) * 1000.0,
        "throughput_inferences_per_sec": max(1, args.iterations) / elapsed if elapsed > 0 else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
