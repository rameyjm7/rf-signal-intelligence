#!/usr/bin/env python3
"""Capture IQ from sdr-gateway and classify it with the shared NoisyDroneRF framer."""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return (arr[:, 0].astype(np.float32) + 1j * arr[:, 1].astype(np.float32)).astype(np.complex64)
    return np.asarray(arr, dtype=np.complex64).reshape(-1)


def selected_iq_window(iq: np.ndarray, payload: dict, window_samples: int) -> np.ndarray:
    start = int((payload.get("scan") or {}).get("selected_start", 0) or 0)
    window = np.asarray(iq)[start : start + int(window_samples)]
    if window.shape[0] < int(window_samples):
        window = np.pad(window, ((0, int(window_samples) - window.shape[0]), (0, 0)), mode="constant")
    return window


def spectral_quality(iq: np.ndarray) -> dict[str, float]:
    complex_iq = iq_to_complex(iq)
    if complex_iq.size < 1024:
        return {
            "snr_db": 0.0,
            "peak_over_floor_db": 0.0,
            "occupied_fraction": 0.0,
            "power_db": -120.0,
        }
    nfft = int(min(16384, 2 ** int(np.floor(np.log2(complex_iq.size)))))
    nfft = max(1024, nfft)
    segment = complex_iq[-nfft:]
    spectrum = np.fft.fftshift(np.fft.fft(segment * np.hanning(nfft).astype(np.float32), n=nfft))
    power_db = 20.0 * np.log10(np.abs(spectrum).astype(np.float64) + 1e-12)
    floor_db = float(np.median(power_db))
    peak_db = float(np.percentile(power_db, 99.7))
    occupied_fraction = float(np.mean(power_db > floor_db + 10.0))
    iq_power_db = float(10.0 * np.log10(np.mean(np.square(np.abs(complex_iq))) + 1e-12))
    return {
        "snr_db": float(peak_db - floor_db),
        "peak_over_floor_db": float(np.max(power_db) - floor_db),
        "occupied_fraction": occupied_fraction,
        "power_db": iq_power_db,
    }


def apply_snr_gate(payload: dict, quality: dict[str, float], *, min_snr_db: float, min_occupied: float) -> dict:
    gated = dict(payload)
    gated["quality"] = quality
    if min_snr_db <= 0.0:
        gated["snr_gate"] = {"passed": True, "reason": "disabled"}
        return gated
    snr_db = float(quality.get("snr_db", 0.0))
    occupied = float(quality.get("occupied_fraction", 0.0))
    passed = snr_db >= min_snr_db and occupied >= min_occupied
    gated["snr_gate"] = {
        "passed": bool(passed),
        "snr_db": snr_db,
        "min_snr_db": float(min_snr_db),
        "occupied_fraction": occupied,
        "min_occupied_fraction": float(min_occupied),
    }
    if passed:
        return gated
    gated["prediction_before_snr_gate"] = gated.get("prediction")
    gated["confidence_before_snr_gate"] = gated.get("confidence")
    gated["prediction"] = "Noise"
    gated["confidence"] = float(gated.get("raw_confidence") or 0.0) if gated.get("raw_prediction") == "Noise" else 0.0
    gated["decision"] = "snr_gate_suppressed"
    return gated


def apply_power_gate(
    payload: dict,
    quality: dict[str, float],
    *,
    noise_floor_db: float | None,
    min_power_snr_db: float,
) -> dict:
    gated = dict(payload)
    if noise_floor_db is None or min_power_snr_db <= 0.0:
        gated["power_gate"] = {"passed": True, "reason": "disabled"}
        return gated
    power_db = float(quality.get("power_db", -120.0))
    power_snr_db = power_db - float(noise_floor_db)
    passed = power_snr_db >= min_power_snr_db
    gated["power_gate"] = {
        "passed": bool(passed),
        "power_db": power_db,
        "noise_floor_db": float(noise_floor_db),
        "power_snr_db": float(power_snr_db),
        "min_power_snr_db": float(min_power_snr_db),
    }
    if passed:
        return gated
    gated["prediction_before_power_gate"] = gated.get("prediction")
    gated["confidence_before_power_gate"] = gated.get("confidence")
    gated["prediction"] = "Noise"
    gated["confidence"] = float(gated.get("raw_confidence") or 0.0) if gated.get("raw_prediction") == "Noise" else 0.0
    gated["decision"] = "power_gate_suppressed"
    return gated


def print_live_table(payload: dict) -> None:
    print_table(payload)
    quality = payload.get("quality") or {}
    snr_gate = payload.get("snr_gate") or {}
    power_gate = payload.get("power_gate") or {}
    print(
        "  quality    : "
        f"spectral_snr={float(quality.get('snr_db', 0.0)):.1f} dB "
        f"occupied={float(quality.get('occupied_fraction', 0.0)):.4f} "
        f"power={float(quality.get('power_db', -120.0)):.1f} dB",
        flush=True,
    )
    if power_gate:
        if power_gate.get("reason") == "disabled":
            print("  power gate : disabled", flush=True)
        else:
            print(
                "  power gate : "
                f"{'pass' if power_gate.get('passed') else 'block'} "
                f"delta={float(power_gate.get('power_snr_db', 0.0)):.1f} dB "
                f"floor={float(power_gate.get('noise_floor_db', -120.0)):.1f} dB "
                f"need={float(power_gate.get('min_power_snr_db', 0.0)):.1f} dB",
                flush=True,
            )
    if snr_gate:
        print(
            "  snr gate   : "
            f"{'pass' if snr_gate.get('passed') else 'block'} "
            f"need={float(snr_gate.get('min_snr_db', 0.0)):.1f} dB",
            flush=True,
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
    parser.add_argument("--lna-gain", type=int, default=24)
    parser.add_argument("--vga-gain", type=int, default=20)
    parser.add_argument("--rx-channel", type=int, default=0)
    parser.add_argument("--iq-format", default="i8", choices=("i8", "cs16", "cf32", "native"))
    parser.add_argument("--capture-samples", type=int, default=4_194_304)
    parser.add_argument(
        "--discard-captures",
        type=int,
        default=1,
        help="Drop this many captures after opening the stream to avoid stale gateway frames.",
    )
    parser.add_argument("--save-iq", type=Path)
    parser.add_argument("--continuous", action="store_true", help="Keep capturing and classifying until Ctrl+C.")
    parser.add_argument("--interval-sec", type=float, default=0.0, help="Delay between reports in --continuous mode.")
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
        default="raw",
    )
    parser.add_argument(
        "--decision-mode",
        choices=("hybrid", "raw", "non-noise"),
        default="hybrid",
    )
    parser.add_argument("--non-noise-threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-snr-db", type=float, default=0.0, help="Suppress non-noise reports below this spectral SNR. Use 0 to disable.")
    parser.add_argument("--min-occupied-fraction", type=float, default=0.002, help="Minimum occupied spectral bins above floor+10 dB.")
    parser.add_argument("--calibration-captures", type=int, default=0, help="Initial captures used to learn idle RX power. Use 0 to disable.")
    parser.add_argument("--min-power-snr-db", type=float, default=0.0, help="Require capture power above calibrated floor by this many dB. Use 0 to disable.")
    parser.add_argument(
        "--signal-present",
        action="store_true",
        help="Allow hybrid mode to promote raw Noise into a non-noise class after SNR gating. Usually leave this off for live monitoring.",
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
        classifier = NoisyDroneFrameClassifier(predictor, labels=load_labels(args.labels), config=config)
        with GatewayIqSource(gateway_config) as source:
            discard_captures = max(0, int(args.discard_captures))
            for idx in range(discard_captures):
                source.read_iq_pairs(args.capture_samples)
                print(
                    f"Discarded startup capture {idx + 1}/{discard_captures}",
                    file=sys.stderr,
                    flush=True,
                )
            noise_floor_db = None
            calibration_captures = max(0, int(args.calibration_captures))
            if calibration_captures:
                powers = []
                print(f"Calibrating idle RX power for {calibration_captures} capture(s)...", file=sys.stderr, flush=True)
                for idx in range(calibration_captures):
                    calibration_iq = source.read_iq_pairs(args.capture_samples)
                    quality = spectral_quality(calibration_iq)
                    powers.append(float(quality["power_db"]))
                    print(
                        f"  calibration {idx + 1}/{calibration_captures}: power={quality['power_db']:.1f} dB",
                        file=sys.stderr,
                        flush=True,
                    )
                noise_floor_db = float(np.median(powers)) if powers else None
                print(f"Calibrated idle RX power floor: {noise_floor_db:.1f} dB", file=sys.stderr, flush=True)
            report_index = 0
            while True:
                iq = source.read_iq_pairs(args.capture_samples)
                if args.save_iq is not None:
                    args.save_iq.parent.mkdir(parents=True, exist_ok=True)
                    np.save(args.save_iq, iq)
                payload = classifier.classify_iq(
                    iq,
                    target_label=args.target_class,
                    signal_present=False,
                )
                quality = spectral_quality(selected_iq_window(iq, payload, args.window_samples))
                signal_present = bool(args.signal_present) and quality["snr_db"] >= float(args.min_snr_db)
                if signal_present:
                    payload = classifier.classify_iq(
                        iq,
                        target_label=args.target_class,
                        signal_present=True,
                    )
                    quality = spectral_quality(selected_iq_window(iq, payload, args.window_samples))
                payload = apply_snr_gate(
                    payload,
                    quality,
                    min_snr_db=float(args.min_snr_db),
                    min_occupied=float(args.min_occupied_fraction),
                )
                payload = apply_power_gate(
                    payload,
                    quality,
                    noise_floor_db=noise_floor_db,
                    min_power_snr_db=float(args.min_power_snr_db),
                )
                payload["gateway"] = {
                    "url": args.gateway_url,
                    "device_id": args.device_id,
                    "freq": float(args.freq),
                    "sample_rate": float(args.sample_rate),
                    "bandwidth": float(args.bandwidth),
                    "iq_format": args.iq_format,
                    "lna_gain": int(args.lna_gain),
                    "vga_gain": int(args.vga_gain),
                    "capture_samples": int(args.capture_samples),
                    "saved_iq": str(args.save_iq) if args.save_iq else None,
                    "report_index": int(report_index),
                    "reported_at": time.time(),
                }
                if report_index:
                    print("", flush=True)
                if args.format == "json":
                    print(json.dumps(payload, indent=2), flush=True)
                else:
                    print_live_table(payload)
                    sys.stdout.flush()
                report_index += 1
                if not args.continuous:
                    break
                if args.interval_sec > 0:
                    time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr, flush=True)
    finally:
        close = getattr(predictor, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
