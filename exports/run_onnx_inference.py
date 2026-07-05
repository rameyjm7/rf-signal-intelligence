#!/usr/bin/env python3
"""Run local ONNX Runtime classification for an exported RF model."""

from __future__ import annotations

import argparse
import json
import random
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
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=1,
        help="ONNX Runtime intra-op thread count. Default 1 avoids Jetson CPU affinity issues.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=1,
        help="ONNX Runtime inter-op thread count. Default 1 avoids Jetson CPU affinity issues.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Output JSON for scripts or a human-readable table.",
    )
    parser.add_argument(
        "--class-sweep",
        action="store_true",
        help="Run one or more NoisyDroneRFv2 dataset samples per class.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="/data/rameyjm7/datasets/NoisyDroneRFv2",
        help="NoisyDroneRFv2 root used with --class-sweep.",
    )
    parser.add_argument(
        "--classes",
        help="Comma-separated class names for --class-sweep; defaults to all labels.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="Number of dataset samples to classify per class in --class-sweep.",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        help="Optional total class-sweep prediction cap; useful for all classes plus a few extras.",
    )
    parser.add_argument(
        "--min-snr",
        type=int,
        default=20,
        help="Minimum dataset SNR for --class-sweep sample selection.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for class-sweep sample selection.")
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


def load_model_input(args: argparse.Namespace, *, iq_file: str | Path | None = None) -> tuple[np.ndarray, dict | None]:
    selected_iq_file = iq_file or args.iq_file
    if selected_iq_file is None:
        x = np.load(args.input).astype(np.float32)
        if x.ndim == 3:
            x = x[None, ...]
        return x, None

    from rf_signal_intelligence.features.spectrogram import (
        SpectrogramConfig,
        iq_to_full_complex_spectrogram,
    )
    from rf_signal_intelligence.live_noisy_drone_rf_classifier import load_iq_file

    iq = load_iq_file(Path(selected_iq_file))
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
        "iq_file": str(selected_iq_file),
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
    *,
    target_class: str | None = None,
    window_score_mode: str | None = None,
) -> tuple[np.ndarray, dict | None]:
    if metadata is None:
        return np.asarray(probs_batch[0], dtype=np.float64), None

    score_mode = window_score_mode or args.window_score_mode
    target_class = target_class or args.target_class
    if score_mode == "auto":
        score_mode = "target" if target_class else "non-noise"

    target_idx = labels.index(target_class) if target_class in labels else None
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
            score_detail = {"target_class": target_class, "target_conditional_confidence": score}
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


def top_entries(probs: np.ndarray, labels: list[str], top_k: int) -> list[dict]:
    ranking = np.argsort(probs)[::-1][: max(1, top_k)]
    return [
        {
            "label": top_label(labels, int(idx)),
            "confidence": float(probs[int(idx)]),
        }
        for idx in ranking
    ]


def run_prediction(
    session,
    input_name: str,
    labels: list[str],
    args: argparse.Namespace,
    *,
    iq_file: str | Path | None = None,
    target_class: str | None = None,
    expected_class: str | None = None,
    decision_mode: str | None = None,
    window_score_mode: str | None = None,
) -> dict:
    x, metadata = load_model_input(args, iq_file=iq_file)
    probs_batch = session.run(None, {input_name: x})[0]

    start = time.perf_counter()
    for _ in range(max(1, args.iterations)):
        probs_batch = session.run(None, {input_name: x})[0]
    elapsed = time.perf_counter() - start

    probs, scan = select_output(
        probs_batch,
        labels,
        metadata,
        args,
        target_class=target_class,
        window_score_mode=window_score_mode,
    )
    raw_ranking = np.argsort(probs)[::-1]
    raw_idx = int(raw_ranking[0])
    resolved_decision_mode = decision_mode or args.decision_mode
    decision_idx, decision_detail = choose_decision(probs, labels, resolved_decision_mode)
    payload: dict = {
        "prediction": top_label(labels, decision_idx),
        "confidence": float(probs[decision_idx]),
        "raw_prediction": top_label(labels, raw_idx),
        "raw_confidence": float(probs[raw_idx]),
        "decision": decision_detail,
        "decision_mode": resolved_decision_mode,
        "scan": scan,
        "top": top_entries(probs, labels, args.top_k),
        "input_shape": list(x.shape),
        "providers": session.get_providers(),
        "iterations": int(args.iterations),
        "avg_latency_ms": elapsed / max(1, args.iterations) * 1000.0,
    }
    if expected_class:
        payload["expected_class"] = expected_class
        payload["correct"] = payload["prediction"] == expected_class
    return payload


def parse_classes(value: str | None, labels: list[str]) -> list[str]:
    if not value:
        return labels
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in labels]
    if unknown:
        raise ValueError(f"Unknown class name(s): {', '.join(unknown)}. Choose from {', '.join(labels)}")
    return requested


def choose_sweep_samples(
    dataset_dir: str | Path,
    labels: list[str],
    classes: list[str],
    *,
    min_snr: int,
    samples_per_class: int,
    max_predictions: int | None,
    seed: int,
) -> list[tuple[str, Path, int]]:
    from rf_signal_intelligence.data.noisy_drone import build_manifest

    records = build_manifest(dataset_dir, min_snr_db=min_snr)
    if not records:
        raise FileNotFoundError(f"No NoisyDroneRFv2 samples with SNR >= {min_snr} found under {dataset_dir}")

    by_class: dict[str, list] = {class_name: [] for class_name in classes}
    for record in records:
        if record.target_raw < 0 or record.target_raw >= len(labels):
            continue
        class_name = labels[record.target_raw]
        if class_name in by_class:
            by_class[class_name].append(record)

    rng = random.Random(seed)
    shuffled_by_class = {}
    for class_name, group in by_class.items():
        shuffled = sorted(group, key=lambda record: record.filepath.name)
        rng.shuffle(shuffled)
        shuffled_by_class[class_name] = shuffled

    selected: list[tuple[str, Path, int]] = []
    for class_name in classes:
        group = shuffled_by_class.get(class_name, [])
        if not group:
            raise FileNotFoundError(
                f"No NoisyDroneRFv2 samples found for class {class_name!r} with SNR >= {min_snr}"
            )
        take = max(1, int(samples_per_class))
        for record in group[:take]:
            selected.append((class_name, record.filepath, record.snr))

    if max_predictions is None:
        return selected
    if max_predictions <= 0:
        raise ValueError("--max-predictions must be greater than 0")
    if len(selected) >= max_predictions:
        return selected[:max_predictions]

    already_selected = {path for _, path, _ in selected}
    offsets = {class_name: max(1, int(samples_per_class)) for class_name in classes}
    while len(selected) < max_predictions:
        added = False
        for class_name in classes:
            group = shuffled_by_class[class_name]
            offset = offsets[class_name]
            while offset < len(group) and group[offset].filepath in already_selected:
                offset += 1
            offsets[class_name] = offset + 1
            if offset >= len(group):
                continue
            record = group[offset]
            selected.append((class_name, record.filepath, record.snr))
            already_selected.add(record.filepath)
            added = True
            if len(selected) >= max_predictions:
                break
        if not added:
            break
    return selected


def format_top_list(payload: dict) -> str:
    return ", ".join(f"{item['label']}={item['confidence']:.3f}" for item in payload["top"])


def print_single_table(payload: dict) -> None:
    print("Prediction")
    print(f"  prediction : {payload['prediction']} ({payload['confidence']:.3f})")
    print(f"  raw top-1  : {payload['raw_prediction']} ({payload['raw_confidence']:.3f})")
    print(f"  top classes: {format_top_list(payload)}")
    print(f"  latency    : {payload['avg_latency_ms']:.2f} ms")
    if payload.get("scan"):
        scan = payload["scan"]
        print(
            f"  window     : start={scan['selected_start']} "
            f"mode={scan['resolved_window_score_mode']} candidates={len(scan['candidates'])}"
        )


def print_sweep_table(rows: list[dict], *, dataset_dir: str | Path, min_snr: int) -> None:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    print(f"NoisyDroneRFv2 ONNX class sweep: {correct}/{total} exact predictions")
    print(f"Dataset: {dataset_dir}")
    print(f"Minimum SNR: {min_snr} dB")
    print()
    print("| # | Expected | Prediction | Conf | Raw top-1 | SNR | Start | File | Top classes |")
    print("|---:|---|---|---:|---|---:|---:|---|---|")
    for idx, row in enumerate(rows, start=1):
        scan = row.get("scan") or {}
        print(
            f"| {idx} | {row['expected_class']} | {row['prediction']} | "
            f"{row['confidence']:.3f} | {row['raw_prediction']} ({row['raw_confidence']:.3f}) | "
            f"{row['snr']} | {scan.get('selected_start', '-')} | "
            f"{Path(row['iq_file']).name} | {format_top_list(row)} |"
        )


def make_session(args: argparse.Namespace):
    import onnxruntime as ort

    requested = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    available = set(ort.get_available_providers())
    providers = [provider for provider in requested if provider in available]
    if not providers:
        raise RuntimeError(f"Requested providers are unavailable. Available providers: {sorted(available)}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.intra_op_threads
    session_options.inter_op_num_threads = args.inter_op_threads
    return ort.InferenceSession(args.onnx, sess_options=session_options, providers=providers)


def main() -> int:
    args = parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    session = make_session(args)
    input_name = session.get_inputs()[0].name

    if args.class_sweep:
        classes = parse_classes(args.classes, labels)
        samples = choose_sweep_samples(
            args.dataset_dir,
            labels,
            classes,
            min_snr=args.min_snr,
            samples_per_class=args.samples_per_class,
            max_predictions=args.max_predictions,
            seed=args.seed,
        )
        rows = []
        for expected_class, iq_file, snr in samples:
            noise_target = expected_class == "Noise"
            payload = run_prediction(
                session,
                input_name,
                labels,
                args,
                iq_file=iq_file,
                target_class=None if noise_target else expected_class,
                expected_class=expected_class,
                decision_mode="raw" if noise_target else args.decision_mode,
                window_score_mode="raw" if noise_target else None,
            )
            payload["iq_file"] = str(iq_file)
            payload["snr"] = snr
            rows.append(payload)
        if args.format == "table":
            print_sweep_table(rows, dataset_dir=args.dataset_dir, min_snr=args.min_snr)
        else:
            print(json.dumps({"results": rows}, indent=2))
        return 0

    payload = run_prediction(session, input_name, labels, args, target_class=args.target_class)
    if args.format == "table":
        print_single_table(payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
