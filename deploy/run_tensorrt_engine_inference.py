#!/usr/bin/env python3
"""Run direct TensorRT engine inference for an exported RF classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        default="models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine",
        help="TensorRT engine built on the target Jetson.",
    )
    parser.add_argument(
        "--sample",
        default="models/noisy_drone_rf_v2/sample_input.npy",
        help="Preprocessed model input saved as .npy.",
    )
    parser.add_argument("--labels", default="models/noisy_drone_rf_v2/labels.json")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--format", choices=("json", "table"), default="table")
    parser.add_argument("--iq-file", help="Raw IQ file to window, spectrogram, scan, and classify.")
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
    parser.add_argument("--classes", help="Comma-separated class names for --class-sweep; defaults to all labels.")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--max-predictions", type=int)
    parser.add_argument("--min-snr", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
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
    )
    parser.add_argument(
        "--decision-mode",
        choices=("raw", "non-noise"),
        default="non-noise",
    )
    return parser.parse_args()


def check_cuda(result, action: str):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA {action} failed: {err}")
    return result[1:] if len(result) > 1 else None


def load_engine(engine_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
    return engine


def tensor_names(engine) -> tuple[list[str], list[str]]:
    import tensorrt as trt

    inputs: list[str] = []
    outputs: list[str] = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(name)
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one input tensor, found {inputs}")
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one output tensor, found {outputs}")
    return inputs, outputs


def shape_has_dynamic_dim(shape: tuple[int, ...]) -> bool:
    return any(dim < 0 for dim in shape)


class TensorRtRunner:
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        try:
            from cuda import cudart
        except ImportError:
            from cuda.bindings import runtime as cudart

        self.cudart = cudart
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        input_names, output_names = tensor_names(self.engine)
        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.stream = check_cuda(cudart.cudaStreamCreate(), "stream creation")[0]

        self.input_device = None
        self.output_device = None
        self.input_bytes = 0
        self.output_bytes = 0
        self.output: np.ndarray | None = None
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

    def prepare(self, sample: np.ndarray) -> None:
        sample = np.ascontiguousarray(sample.astype(np.float32))
        if shape_has_dynamic_dim(tuple(self.engine.get_tensor_shape(self.input_name))):
            if not self.context.set_input_shape(self.input_name, sample.shape):
                raise RuntimeError(f"TensorRT rejected input shape {sample.shape} for {self.input_name}.")

        input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if tuple(sample.shape) != input_shape:
            raise RuntimeError(f"Sample shape {sample.shape} does not match engine input shape {input_shape}.")

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output = np.empty(output_shape, dtype=self.output_dtype)
        input_bytes = int(sample.nbytes)
        output_bytes = int(output.nbytes)

        if self.input_device is not None and input_bytes == self.input_bytes and output_bytes == self.output_bytes:
            self.output = output
            return

        self.free_buffers()
        self.input_device = check_cuda(self.cudart.cudaMalloc(input_bytes), "input allocation")[0]
        self.output_device = check_cuda(self.cudart.cudaMalloc(output_bytes), "output allocation")[0]
        self.input_bytes = input_bytes
        self.output_bytes = output_bytes
        self.output = output
        if not self.context.set_tensor_address(self.input_name, int(self.input_device)):
            raise RuntimeError(f"Could not bind input tensor {self.input_name}.")
        if not self.context.set_tensor_address(self.output_name, int(self.output_device)):
            raise RuntimeError(f"Could not bind output tensor {self.output_name}.")

    def infer(self, sample: np.ndarray) -> np.ndarray:
        sample = np.ascontiguousarray(sample.astype(np.float32))
        self.prepare(sample)
        assert self.output is not None
        assert self.input_device is not None
        assert self.output_device is not None
        check_cuda(
            self.cudart.cudaMemcpyAsync(
                self.input_device,
                sample.ctypes.data,
                self.input_bytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "host-to-device copy",
        )
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execution failed.")
        check_cuda(
            self.cudart.cudaMemcpyAsync(
                self.output.ctypes.data,
                self.output_device,
                self.output_bytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "device-to-host copy",
        )
        check_cuda(self.cudart.cudaStreamSynchronize(self.stream), "stream synchronize")
        return np.asarray(self.output, dtype=np.float32).copy()

    def free_buffers(self) -> None:
        if self.input_device is not None:
            check_cuda(self.cudart.cudaFree(self.input_device), "input free")
            self.input_device = None
        if self.output_device is not None:
            check_cuda(self.cudart.cudaFree(self.output_device), "output free")
            self.output_device = None

    def close(self) -> None:
        self.free_buffers()
        check_cuda(self.cudart.cudaStreamDestroy(self.stream), "stream destroy")

    @property
    def metadata(self) -> dict:
        return {
            "input_name": self.input_name,
            "input_shape": list(self.context.get_tensor_shape(self.input_name)),
            "output_name": self.output_name,
            "output_shape": list(self.context.get_tensor_shape(self.output_name)),
        }


def run_engine(engine_path: Path, sample: np.ndarray, iterations: int) -> tuple[np.ndarray, dict]:
    import tensorrt as trt
    try:
        from cuda import cudart
    except ImportError:
        from cuda.bindings import runtime as cudart

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create TensorRT execution context.")

    input_names, output_names = tensor_names(engine)
    input_name = input_names[0]
    output_name = output_names[0]

    sample = np.ascontiguousarray(sample.astype(np.float32))
    if shape_has_dynamic_dim(tuple(engine.get_tensor_shape(input_name))):
        if not context.set_input_shape(input_name, sample.shape):
            raise RuntimeError(f"TensorRT rejected input shape {sample.shape} for {input_name}.")

    input_shape = tuple(context.get_tensor_shape(input_name))
    if tuple(sample.shape) != input_shape:
        raise RuntimeError(f"Sample shape {sample.shape} does not match engine input shape {input_shape}.")

    output_shape = tuple(context.get_tensor_shape(output_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output = np.empty(output_shape, dtype=output_dtype)

    stream = check_cuda(cudart.cudaStreamCreate(), "stream creation")[0]
    input_bytes = int(sample.nbytes)
    output_bytes = int(output.nbytes)
    input_device = check_cuda(cudart.cudaMalloc(input_bytes), "input allocation")[0]
    output_device = check_cuda(cudart.cudaMalloc(output_bytes), "output allocation")[0]

    try:
        if not context.set_tensor_address(input_name, int(input_device)):
            raise RuntimeError(f"Could not bind input tensor {input_name}.")
        if not context.set_tensor_address(output_name, int(output_device)):
            raise RuntimeError(f"Could not bind output tensor {output_name}.")

        def infer_once() -> np.ndarray:
            check_cuda(
                cudart.cudaMemcpyAsync(
                    input_device,
                    sample.ctypes.data,
                    input_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                "host-to-device copy",
            )
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("TensorRT execution failed.")
            check_cuda(
                cudart.cudaMemcpyAsync(
                    output.ctypes.data,
                    output_device,
                    output_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
                "device-to-host copy",
            )
            check_cuda(cudart.cudaStreamSynchronize(stream), "stream synchronize")
            return np.asarray(output, dtype=np.float32).copy()

        probs = infer_once()
        start = time.perf_counter()
        for _ in range(max(1, iterations)):
            probs = infer_once()
        elapsed = time.perf_counter() - start
    finally:
        check_cuda(cudart.cudaFree(input_device), "input free")
        check_cuda(cudart.cudaFree(output_device), "output free")
        check_cuda(cudart.cudaStreamDestroy(stream), "stream destroy")

    return probs, {
        "input_name": input_name,
        "input_shape": list(input_shape),
        "output_name": output_name,
        "output_shape": list(output_shape),
        "iterations": int(iterations),
        "avg_latency_ms": elapsed / max(1, iterations) * 1000.0,
        "throughput_inferences_per_sec": max(1, iterations) / elapsed if elapsed > 0 else None,
    }


def run_tensor_batch(runner: TensorRtRunner, x: np.ndarray, iterations: int) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    outputs = []
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        outputs = [runner.infer(x[idx : idx + 1])[0] for idx in range(x.shape[0])]
    elapsed = time.perf_counter() - start
    return np.stack(outputs, axis=0), {
        **runner.metadata,
        "iterations": int(iterations),
        "windows_per_iteration": int(x.shape[0]),
        "avg_latency_ms": elapsed / max(1, iterations * x.shape[0]) * 1000.0,
        "throughput_inferences_per_sec": max(1, iterations * x.shape[0]) / elapsed if elapsed > 0 else None,
    }


def label_for(labels: list[str], idx: int) -> str:
    return labels[idx] if idx < len(labels) else str(idx)


def summarize(probs: np.ndarray, labels: list[str], top_k: int, metadata: dict) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim > 1:
        probs = probs[0]
    ranking = np.argsort(probs)[::-1][: max(1, top_k)]
    top = [{"label": label_for(labels, int(idx)), "confidence": float(probs[int(idx)])} for idx in ranking]
    best = top[0]
    return {
        "prediction": best["label"],
        "confidence": best["confidence"],
        "top": top,
        **metadata,
    }


def print_table(result: dict) -> None:
    print("TensorRT Prediction")
    print(f"  prediction : {result['prediction']} ({result['confidence']:.3f})")
    print("  top classes: " + ", ".join(f"{row['label']}={row['confidence']:.3f}" for row in result["top"]))
    print(f"  latency    : {result['avg_latency_ms']:.2f} ms")
    print(f"  throughput : {result['throughput_inferences_per_sec']:.2f} infer/sec")
    print(f"  input      : {result['input_name']} {result['input_shape']}")
    print(f"  output     : {result['output_name']} {result['output_shape']}")


def run_prediction(
    runner: TensorRtRunner,
    labels: list[str],
    args: argparse.Namespace,
    *,
    iq_file: str | Path | None = None,
    target_class: str | None = None,
    expected_class: str | None = None,
    decision_mode: str | None = None,
    window_score_mode: str | None = None,
) -> dict:
    from exports.run_onnx_inference import (
        choose_decision,
        load_model_input,
        select_output,
        top_entries,
        top_label,
    )

    x, metadata = load_model_input(args, iq_file=iq_file)
    probs_batch, perf = run_tensor_batch(runner, x, args.iterations)
    probs, scan = select_output(
        probs_batch,
        labels,
        metadata,
        args,
        target_class=target_class,
        window_score_mode=window_score_mode,
    )
    raw_idx = int(np.argmax(probs))
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
        **perf,
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
    selected: list[tuple[str, Path, int]] = []
    for class_name in classes:
        group = sorted(by_class.get(class_name, []), key=lambda record: record.filepath.name)
        rng.shuffle(group)
        if not group:
            raise FileNotFoundError(f"No samples found for class {class_name!r} with SNR >= {min_snr}")
        for record in group[: max(1, samples_per_class)]:
            selected.append((class_name, record.filepath, record.snr))
    return selected[:max_predictions] if max_predictions else selected


def format_top_list(payload: dict) -> str:
    return ", ".join(f"{item['label']}={item['confidence']:.3f}" for item in payload["top"])


def print_prediction_table(payload: dict) -> None:
    print("TensorRT Prediction")
    if payload.get("expected_class"):
        print(f"  expected   : {payload['expected_class']} correct={payload['correct']}")
    print(f"  prediction : {payload['prediction']} ({payload['confidence']:.3f})")
    print(f"  raw top-1  : {payload['raw_prediction']} ({payload['raw_confidence']:.3f})")
    print(f"  top classes: {format_top_list(payload)}")
    print(f"  latency    : {payload['avg_latency_ms']:.2f} ms/window")
    if payload.get("scan"):
        scan = payload["scan"]
        print(
            f"  window     : start={scan['selected_start']} "
            f"mode={scan['resolved_window_score_mode']} candidates={len(scan['candidates'])}"
        )


def print_sweep_table(rows: list[dict], *, dataset_dir: str | Path, min_snr: int) -> None:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    print(f"NoisyDroneRFv2 TensorRT class sweep: {correct}/{total} exact predictions")
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


def main() -> int:
    args = parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    runner = TensorRtRunner(Path(args.engine))
    try:
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
                    runner,
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
            if args.format == "json":
                print(json.dumps({"results": rows}, indent=2))
            else:
                print_sweep_table(rows, dataset_dir=args.dataset_dir, min_snr=args.min_snr)
            return 0

        if args.iq_file:
            payload = run_prediction(runner, labels, args, target_class=args.target_class)
            if args.format == "json":
                print(json.dumps(payload, indent=2))
            else:
                print_prediction_table(payload)
            return 0

        sample = np.load(args.sample)
        probs, metadata = run_tensor_batch(runner, sample, args.iterations)
        result = summarize(probs, labels, args.top_k, metadata)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print_table(result)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
