#!/usr/bin/env python3
"""Run local ONNX Runtime classification for an exported RF model."""

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
    parser.add_argument("--input", default="models/noisy_drone_rf_v2/sample_input.npy")
    parser.add_argument("--labels", default="models/noisy_drone_rf_v2/labels.json")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x = np.load(args.input).astype(np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    import onnxruntime as ort

    requested = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    available = set(ort.get_available_providers())
    providers = [provider for provider in requested if provider in available]
    if not providers:
        raise RuntimeError(f"Requested providers are unavailable. Available providers: {sorted(available)}")

    session = ort.InferenceSession(args.onnx, providers=providers)
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: x})[0]

    start = time.perf_counter()
    for _ in range(max(1, args.iterations)):
        probs = session.run(None, {input_name: x})[0]
    elapsed = time.perf_counter() - start

    probs = np.asarray(probs[0], dtype=np.float64)
    ranking = np.argsort(probs)[::-1][: max(1, args.top_k)]
    payload = {
        "prediction": labels[int(ranking[0])] if int(ranking[0]) < len(labels) else str(int(ranking[0])),
        "confidence": float(probs[int(ranking[0])]),
        "top": [
            {
                "label": labels[int(idx)] if int(idx) < len(labels) else str(int(idx)),
                "confidence": float(probs[int(idx)]),
            }
            for idx in ranking
        ],
        "input_shape": list(x.shape),
        "providers": session.get_providers(),
        "iterations": int(args.iterations),
        "avg_latency_ms": elapsed / max(1, args.iterations) * 1000.0,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
