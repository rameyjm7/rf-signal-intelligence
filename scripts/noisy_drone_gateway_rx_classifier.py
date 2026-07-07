#!/usr/bin/env python3
"""Capture IQ from sdr-gateway and classify it with the shared NoisyDroneRF framer."""

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

from noisy_drone_frame_classifier import load_labels, make_predictor, print_table  # noqa: E402
from rf_signal_intelligence.gateway_iq import GatewayIqSource, GatewayStreamConfig  # noqa: E402
from rf_signal_intelligence.live_noisy_drone_rf_classifier import LABEL_NAMES  # noqa: E402
from rf_signal_intelligence.noisy_drone_framing import (  # noqa: E402
    NoisyDroneFrameClassifier,
    NoisyDroneFrameConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    parser.add_argument("--gateway-token")
    parser.add_argument("--device-id", default="bladerf:0")
    parser.add_argument("--freq", type=float, default=2_470_000_000.0)
    parser.add_argument("--sample-rate", type=float, default=20_000_000.0)
    parser.add_argument("--bandwidth", type=float, default=20_000_000.0)
    parser.add_argument("--lna-gain", type=int, default=32)
    parser.add_argument("--vga-gain", type=int, default=40)
    parser.add_argument("--rx-channel", type=int, default=0)
    parser.add_argument("--iq-format", default="i8", choices=("i8", "cs16", "cf32", "native"))
    parser.add_argument("--capture-samples", type=int, default=4_194_304)
    parser.add_argument("--save-iq", type=Path)
    parser.add_argument("--backend", choices=("keras", "tensorrt"), default="keras")
    parser.add_argument("--model", type=Path, default=Path("models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras"))
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--target-class", choices=LABEL_NAMES)
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
    parser.add_argument(
        "--signal-present",
        action="store_true",
        help="Allow hybrid mode to promote raw Noise into a non-noise class.",
    )
    parser.add_argument("--format", choices=("json", "table"), default="table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gateway_config = GatewayStreamConfig(
        base_url=args.gateway_url,
        device_id=args.device_id,
        center_freq_hz=int(round(args.freq)),
        sample_rate_sps=int(round(args.sample_rate)),
        bandwidth_hz=int(round(args.bandwidth)),
        lna_gain_db=args.lna_gain,
        vga_gain_db=args.vga_gain,
        rx_channel=args.rx_channel,
        iq_format="i8" if args.iq_format == "native" else args.iq_format,
        token=args.gateway_token,
    )
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
        with GatewayIqSource(gateway_config) as source:
            iq = source.read_iq_pairs(args.capture_samples)
        if args.save_iq is not None:
            args.save_iq.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_iq, iq)
        classifier = NoisyDroneFrameClassifier(predictor, labels=load_labels(args.labels), config=config)
        payload = classifier.classify_iq(
            iq,
            target_label=args.target_class,
            signal_present=bool(args.signal_present),
        )
        payload["gateway"] = {
            "url": args.gateway_url,
            "device_id": args.device_id,
            "freq": float(args.freq),
            "sample_rate": float(args.sample_rate),
            "bandwidth": float(args.bandwidth),
            "capture_samples": int(args.capture_samples),
            "saved_iq": str(args.save_iq) if args.save_iq else None,
        }
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
