#!/usr/bin/env python3
"""Continuously transmit NoisyDroneRFv2-like IQ from a selected SDR."""

from __future__ import annotations

import argparse
import itertools
import signal
import sys
import time
from pathlib import Path

import numpy as np

from rf_signal_intelligence.live_noisy_drone_rf_classifier import (
    DEFAULT_BANDWIDTH,
    DEFAULT_FREQ,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TX_DATASET_DIR,
    DEFAULT_TX_DEVICE_ARGS,
    LABEL_NAMES,
    SoapyIqSink,
    choose_tx_sample,
    describe_noisy_drone_sample,
    frontend_label,
    generate_gan_iq,
    load_iq_file,
    prepare_tx_iq,
    resolve_gan_generator_path,
)


def parse_class_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return [name for name in LABEL_NAMES if name != "Noise"]
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise argparse.ArgumentTypeError("At least one class is required.")
    invalid = [item for item in classes if item not in LABEL_NAMES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown class name(s): {', '.join(invalid)}. Choose from {', '.join(LABEL_NAMES)} or all."
        )
    return classes


def iq_summary(iq: np.ndarray) -> str:
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    peak = float(np.max(np.abs(complex_iq))) if complex_iq.size else 0.0
    power = float(10.0 * np.log10(np.mean(np.abs(complex_iq) ** 2) + 1e-12))
    return f"samples={len(iq)} power={power:.1f} dB peak={peak:.3f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("dataset", "gan", "iq-file"),
        default="gan",
        help="IQ source to transmit.",
    )
    parser.add_argument(
        "--classes",
        type=parse_class_list,
        default=parse_class_list("DJI"),
        help="Comma-separated NoisyDroneRFv2 classes to cycle, or 'all'. Ignored for --source iq-file.",
    )
    parser.add_argument("--iq-file", type=Path, help="IQ file to repeatedly transmit with --source iq-file.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_TX_DATASET_DIR)
    parser.add_argument("--min-snr", type=int, default=20, help="Minimum SNR for dataset samples.")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for repeatable dataset/GAN selection.")
    parser.add_argument("--gan-generator", type=Path, help="Keras generator path for --source gan.")
    parser.add_argument("--gan-samples", type=int, default=256, help="Generated GAN windows per class burst.")
    parser.add_argument("--gan-batch-size", type=int, default=32)
    parser.add_argument("--device-args", default=DEFAULT_TX_DEVICE_ARGS, help="SoapySDR TX device args.")
    parser.add_argument("--channel", type=int, default=0, help="TX channel. For bladeRF, TX1 is channel 0.")
    parser.add_argument("--antenna", default="TX", help="TX antenna name to select when available.")
    parser.add_argument("--freq", type=float, default=DEFAULT_FREQ)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument("--gain", type=float, default=60.0)
    parser.add_argument("--stream-args", help="Soapy TX stream args.")
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--pad-sec", type=float, default=0.005)
    parser.add_argument("--chunk-samples", type=int, default=65536)
    parser.add_argument(
        "--repeat-per-class",
        type=int,
        default=1,
        help="Transmit this many prepared bursts before moving to the next class. Use 0 to repeat forever on the first class.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="For GAN/dataset sources, choose or generate a fresh burst each time a class is revisited.",
    )
    parser.add_argument("--gap-sec", type=float, default=0.0, help="Optional silence/wait between bursts.")
    return parser.parse_args(argv)


def load_generator(path: Path | None):
    from tensorflow.keras.models import load_model

    generator_path = resolve_gan_generator_path(path)
    return generator_path, load_model(generator_path, compile=False)


def build_source_cache(args: argparse.Namespace) -> dict[str, tuple[str, np.ndarray]]:
    if args.source == "iq-file":
        if args.iq_file is None:
            raise ValueError("--iq-file is required with --source iq-file.")
    return {}


def make_burst(
    args: argparse.Namespace,
    *,
    class_name: str,
    cycle_index: int,
    cache: dict[str, tuple[str, np.ndarray]],
    gan_model=None,
    gan_path: Path | None = None,
) -> tuple[str, np.ndarray]:
    cache_key = "iq-file" if args.source == "iq-file" else class_name
    if not args.regenerate and cache_key in cache:
        return cache[cache_key]

    seed = int(args.seed) + int(cycle_index)
    if args.source == "iq-file":
        assert args.iq_file is not None
        source_desc, raw_iq = str(args.iq_file), load_iq_file(args.iq_file)
    elif args.source == "dataset":
        path, raw_iq = choose_tx_sample(
            args.dataset_dir,
            None,
            seed=seed,
            min_snr=args.min_snr,
            target=None,
            class_name=class_name,
        )
        source_desc = describe_noisy_drone_sample(path)
    else:
        if gan_model is None or gan_path is None:
            raise RuntimeError("GAN source selected without a loaded generator.")
        class_idx = LABEL_NAMES.index(class_name)
        raw_iq = generate_gan_iq(
            gan_model,
            class_idx=class_idx,
            seed=seed,
            samples=args.gan_samples,
            batch_size=args.gan_batch_size,
        )
        source_desc = f"GAN synthetic {class_name} from {gan_path.name} windows={args.gan_samples}"

    prepared = prepare_tx_iq(
        raw_iq,
        amplitude=args.amplitude,
        pad_seconds=args.pad_sec,
        sample_rate=args.sample_rate,
    )
    result = source_desc, prepared
    if not args.regenerate:
        cache[cache_key] = result
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    gan_path = None
    gan_model = None
    if args.source == "gan":
        gan_path, gan_model = load_generator(args.gan_generator)
        print(f"Loaded GAN generator: {gan_path}", flush=True)

    cache = build_source_cache(args)
    sink = SoapyIqSink(
        device_args=args.device_args,
        channel=args.channel,
        sample_rate=args.sample_rate,
        freq=args.freq,
        gain=args.gain,
        antenna=args.antenna,
        bandwidth=args.bandwidth,
        stream_args=args.stream_args,
    )
    print(
        f"TX ready with SoapySDR device={args.device_args!r} "
        f"frontend={frontend_label(args.device_args, 'TX', args.channel)} "
        f"antenna={args.antenna!r} freq={args.freq:.0f} Hz "
        f"sample_rate={args.sample_rate:.0f} S/s bandwidth={args.bandwidth:.0f} Hz gain={args.gain}",
        flush=True,
    )
    print("Transmitting until Ctrl-C...", flush=True)

    try:
        cycle_index = 0
        class_iterable = ["iq-file"] if args.source == "iq-file" else args.classes
        for class_name in itertools.cycle(class_iterable):
            if stop:
                break
            source_desc, prepared = make_burst(
                args,
                class_name=class_name,
                cycle_index=cycle_index,
                cache=cache,
                gan_model=gan_model,
                gan_path=gan_path,
            )
            repeats = 1 if args.repeat_per_class > 0 else sys.maxsize
            repeats = min(repeats, args.repeat_per_class) if args.repeat_per_class > 0 else repeats
            for repeat_idx in range(repeats):
                if stop:
                    break
                print(
                    f"TX class={class_name} repeat={repeat_idx + 1}/{repeats if repeats != sys.maxsize else 'inf'} "
                    f"source={source_desc} {iq_summary(prepared)}",
                    flush=True,
                )
                sink.write_iq(prepared, chunk_samples=args.chunk_samples)
                if args.gap_sec > 0 and not stop:
                    time.sleep(args.gap_sec)
            cycle_index += 1
    finally:
        sink.close()
        print("TX stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
