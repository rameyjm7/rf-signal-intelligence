#!/usr/bin/env python3
"""Run local ONNX Runtime classification for an exported RF model."""

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
        "--onnx",
        default="models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx",
    )
    parser.add_argument("--input", default="models/noisy_drone_rf_v2/sample_input.npy")
    parser.add_argument("--iq-file", help="Raw IQ file to window, spectrogram, scan, and classify.")
    parser.add_argument("--labels", default="models/noisy_drone_rf_v2/labels.json")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--window-samples", type=int, default=1_048_576)
    parser.add_argument("--scan-stride-samples", type=int, default=262_144)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=1024)
    parser.add_argument("--time-bins", type=int, default=1024)
    parser.add_argument("--target-class", help="Class to favor when selecting a scanned IQ window.")
    parser.add_argument(
        "--window-score-mode",
        choices=("auto", "target", "non-noise", "raw"),
        default="auto",
        help="How to choose the best scanned IQ window.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=("raw", "non-noise"),
        default="non-noise",
        help="Use raw top-1 output or promote the best non-Noise class.",
    )
    return parser.parse_args()


def top_label(labels: list[str], idx: int) -> str:
    return labels[idx] if idx < len(labels) else str(idx)


def non_noise_stats(probs: np.ndarray, labels: list[str]) -> tuple[int, float, float]:
    masked = probs.copy()
    noise_idx = labels.index("Noise") if "Noise" in labels else None
    if noise_idx is not None:
        masked[noise_idx] = -np.inf
    idx = int(np.argmax(masked))
    non_noise_mass = float(np.sum(probs) - (probs[noise_idx] if noise_idx is not None else 0.0))
    confidence = float(probs[idx] / non_noise_mass) if non_noise_mass > 0.0 else float(probs[idx])
    return idx, confidence, non_noise_mass


def choose_decision(probs: np.ndarray, labels: list[str], decision_mode: str) -> tuple[int, dict | None]:
    raw_idx = int(np.argmax(probs))
    if decision_mode != "non-noise" or "Noise" not in labels or len(labels) <= 1:
        return raw_idx, None
    idx, confidence, non_noise_mass = non_noise_stats(probs, labels)
    return idx, {
        "mode": "non-noise",
        "conditional_confidence": confidence,
        "non_noise_probability_mass": non_noise_mass,
    }


def candidate_window_starts(total_samples: int, window_samples: int, stride_samples: int) -> list[int]:
    if total_samples <= window_samples:
        return [0]
    stride_samples = max(1, int(stride_samples))
    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    final_start = total_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def load_model_input(args: argparse.Namespace) -> tuple[np.ndarray, dict | None]:
    if args.iq_file is None:
        x = np.load(args.input).astype(np.float32)
        if x.ndim == 3:
            x = x[None, ...]
        return x, None

    from rf_signal_intelligence.features.spectrogram import (
        SpectrogramConfig,
        iq_to_full_complex_spectrogram,
    )
    from rf_signal_intelligence.live_noisy_drone_rf_classifier import load_iq_file

    iq = load_iq_file(Path(args.iq_file))
    if iq.shape[0] < args.window_samples:
        iq = np.pad(iq, ((0, args.window_samples - iq.shape[0]), (0, 0)), mode="constant")
    config = SpectrogramConfig(
        sample_len=args.window_samples,
        nfft=args.nfft,
        hop=args.hop,
        time_bins=args.time_bins,
    )
    windows = []
    starts = candidate_window_starts(iq.shape[0], args.window_samples, args.scan_stride_samples)
    for start in starts:
        window = iq[start : start + args.window_samples, :2]
        windows.append(iq_to_full_complex_spectrogram(window, config))
    return np.stack(windows, axis=0).astype(np.float32), {
        "iq_file": str(args.iq_file),
        "raw_iq_samples": int(iq.shape[0]),
        "candidate_starts": starts,
        "window_samples": int(args.window_samples),
        "window_score_mode": args.window_score_mode,
        "target_class": args.target_class,
    }


def select_output(
    probs_batch: np.ndarray,
    labels: list[str],
    metadata: dict | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict | None]:
    if metadata is None:
        return np.asarray(probs_batch[0], dtype=np.float64), None

    score_mode = args.window_score_mode
    if score_mode == "auto":
        score_mode = "target" if args.target_class else "non-noise"

    target_idx = labels.index(args.target_class) if args.target_class in labels else None
    candidates = []
    best_idx = 0
    best_score = -np.inf
    for idx, probs in enumerate(np.asarray(probs_batch, dtype=np.float64)):
        raw_idx = int(np.argmax(probs))
        non_noise_idx, non_noise_confidence, non_noise_mass = non_noise_stats(probs, labels)
        if score_mode == "target" and target_idx is not None:
            score = (
                float(probs[target_idx] / non_noise_mass)
                if non_noise_mass > 0.0
                else float(probs[target_idx])
            )
            score_detail = {"target_class": args.target_class, "target_conditional_confidence": score}
        elif score_mode == "non-noise":
            score = non_noise_confidence
            score_detail = {"non_noise_conditional_confidence": non_noise_confidence}
        else:
            score = float(probs[raw_idx])
            score_detail = {"raw_confidence": score}
        candidate = {
            "start": metadata["candidate_starts"][idx],
            "score": float(score),
            "raw_prediction": top_label(labels, raw_idx),
            "raw_confidence": float(probs[raw_idx]),
            "best_non_noise": top_label(labels, non_noise_idx),
            "best_non_noise_confidence": non_noise_confidence,
            **score_detail,
        }
        candidates.append(candidate)
        if score > best_score:
            best_score = score
            best_idx = idx

    return np.asarray(probs_batch[best_idx], dtype=np.float64), {
        **metadata,
        "selected_start": metadata["candidate_starts"][best_idx],
        "selected_index": int(best_idx),
        "resolved_window_score_mode": score_mode,
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    x, metadata = load_model_input(args)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    import onnxruntime as ort

    requested = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    available = set(ort.get_available_providers())
    providers = [provider for provider in requested if provider in available]
    if not providers:
        raise RuntimeError(f"Requested providers are unavailable. Available providers: {sorted(available)}")

    session = ort.InferenceSession(args.onnx, providers=providers)
    input_name = session.get_inputs()[0].name
    probs_batch = session.run(None, {input_name: x})[0]

    start = time.perf_counter()
    for _ in range(max(1, args.iterations)):
        probs_batch = session.run(None, {input_name: x})[0]
    elapsed = time.perf_counter() - start

    probs, scan = select_output(probs_batch, labels, metadata, args)
    raw_ranking = np.argsort(probs)[::-1]
    raw_idx = int(raw_ranking[0])
    decision_idx, decision_detail = choose_decision(probs, labels, args.decision_mode)
    ranking = raw_ranking[: max(1, args.top_k)]
    payload = {
        "prediction": top_label(labels, decision_idx),
        "confidence": float(probs[decision_idx]),
        "raw_prediction": top_label(labels, raw_idx),
        "raw_confidence": float(probs[raw_idx]),
        "decision": decision_detail,
        "scan": scan,
        "top": [
            {
                "label": top_label(labels, int(idx)),
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
