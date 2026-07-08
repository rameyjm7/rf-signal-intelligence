"""Reusable NoisyDroneRF IQ framing and classification helpers.

This module is the shared seam for live SDR captures, offline files, TensorRT,
and SDR-Shark integration. It intentionally mirrors the proven live classifier
windowing flow: capture more IQ than one model window, classify candidate
windows, and select the best candidate using a target or non-noise score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rf_signal_intelligence.features.spectrogram import (
    SpectrogramConfig,
    iq_to_full_complex_spectrogram,
)
from rf_signal_intelligence.live_noisy_drone_rf_classifier import (
    LABEL_NAMES,
    best_non_noise_prediction,
    candidate_window_starts,
    capture_stats,
    choose_final_prediction,
    coerce_iq_array,
    conditional_class_confidence,
    select_high_power_window,
)

PredictBatch = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NoisyDroneFrameConfig:
    """Configuration for converting raw IQ into NoisyDroneRF model windows."""

    window_samples: int = 1_048_576
    nfft: int = 1024
    hop: int = 1024
    time_bins: int = 1024
    scan_windows: bool = True
    scan_stride_samples: int = 262_144
    burst_smooth_samples: int = 512
    window_score_mode: str = "auto"
    decision_mode: str = "hybrid"
    non_noise_threshold: float = 0.55
    phase_tta: int = 1
    top_k: int = 3

    @property
    def spectrogram(self) -> SpectrogramConfig:
        return SpectrogramConfig(
            sample_len=self.window_samples,
            nfft=self.nfft,
            hop=self.hop,
            time_bins=self.time_bins,
            burst_smooth_samples=self.burst_smooth_samples,
        )


def top_entries(probs: np.ndarray, labels: list[str], top_k: int) -> list[dict[str, float | str]]:
    ranking = np.argsort(probs)[::-1][: max(1, int(top_k))]
    return [
        {
            "label": labels[int(idx)] if int(idx) < len(labels) else str(int(idx)),
            "confidence": float(probs[int(idx)]),
        }
        for idx in ranking
    ]


def _as_probability_batch(value: np.ndarray) -> np.ndarray:
    probs = np.asarray(value, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[None, :]
    if probs.ndim != 2:
        raise ValueError(f"Expected prediction batch with shape (batch, classes), got {probs.shape}")
    return probs


def _score_candidate(
    probs: np.ndarray,
    labels: list[str],
    *,
    score_mode: str,
    target_label: str | None,
) -> tuple[float, dict[str, float | str]]:
    raw_idx = int(np.argmax(probs))
    raw_label = labels[raw_idx] if raw_idx < len(labels) else str(raw_idx)
    raw_confidence = float(probs[raw_idx])
    non_noise_label, non_noise_confidence = best_non_noise_prediction(probs, labels)
    detail: dict[str, float | str] = {
        "raw_prediction": raw_label,
        "raw_confidence": raw_confidence,
        "best_non_noise": non_noise_label,
        "best_non_noise_confidence": float(non_noise_confidence),
    }

    if score_mode == "target" and target_label in labels:
        target_idx = labels.index(target_label)
        score = conditional_class_confidence(probs, labels, target_idx)
        detail["target_class"] = target_label
        detail["target_conditional_confidence"] = float(score)
        return float(score), detail
    if score_mode == "non-noise":
        return float(non_noise_confidence), detail
    return raw_confidence, detail


def build_model_windows(
    iq: np.ndarray,
    config: NoisyDroneFrameConfig,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return model input tensors, candidate starts, and normalized/padded IQ."""

    raw_iq = coerce_iq_array(iq)
    if raw_iq.shape[0] < config.window_samples:
        raw_iq = np.pad(
            raw_iq,
            ((0, config.window_samples - raw_iq.shape[0]), (0, 0)),
            mode="constant",
        )

    if config.scan_windows and raw_iq.shape[0] > config.window_samples:
        starts = candidate_window_starts(
            raw_iq.shape[0],
            config.window_samples,
            config.scan_stride_samples,
        )
    else:
        _, start = select_high_power_window(
            raw_iq,
            window_samples=config.window_samples,
            smooth_samples=config.burst_smooth_samples,
        )
        starts = [start]

    spec_config = config.spectrogram
    windows = [
        iq_to_full_complex_spectrogram(raw_iq[start : start + config.window_samples, :2], spec_config)
        for start in starts
    ]
    return np.stack(windows, axis=0).astype(np.float32), starts, raw_iq


class NoisyDroneFrameClassifier:
    """Classify raw IQ with the NoisyDroneRF framing used by the live script."""

    def __init__(
        self,
        predict_batch: PredictBatch,
        *,
        labels: list[str] | None = None,
        config: NoisyDroneFrameConfig | None = None,
    ) -> None:
        self.predict_batch = predict_batch
        self.labels = list(labels or LABEL_NAMES)
        self.config = config or NoisyDroneFrameConfig()

    def classify_iq(
        self,
        iq: np.ndarray,
        *,
        target_label: str | None = None,
        signal_present: bool = True,
    ) -> dict:
        model_windows, starts, raw_iq = build_model_windows(iq, self.config)
        probs_batch = _as_probability_batch(self.predict_batch(model_windows))
        if probs_batch.shape[0] != len(starts):
            raise ValueError(
                f"Predictor returned {probs_batch.shape[0]} rows for {len(starts)} candidate windows"
            )

        score_mode = self.config.window_score_mode
        if score_mode == "auto":
            score_mode = "target" if target_label else "non-noise"

        best_idx = 0
        best_score = -np.inf
        candidates: list[dict] = []
        for idx, probs in enumerate(probs_batch):
            score, detail = _score_candidate(
                probs,
                self.labels,
                score_mode=score_mode,
                target_label=target_label,
            )
            candidate = {
                "start": int(starts[idx]),
                "score": float(score),
                **detail,
            }
            candidates.append(candidate)
            if score > best_score:
                best_idx = idx
                best_score = score

        selected_start = int(starts[best_idx])
        selected_iq = raw_iq[selected_start : selected_start + self.config.window_samples].astype(
            np.float32,
            copy=False,
        )
        selected_probs = np.asarray(probs_batch[best_idx], dtype=np.float64)
        raw_idx = int(np.argmax(selected_probs))
        raw_label = self.labels[raw_idx] if raw_idx < len(self.labels) else str(raw_idx)
        raw_confidence = float(selected_probs[raw_idx])
        final_label, final_confidence, decision_detail = choose_final_prediction(
            raw_label,
            raw_confidence,
            selected_probs,
            self.labels,
            decision_mode=self.config.decision_mode,
            non_noise_threshold=self.config.non_noise_threshold,
            signal_present=signal_present,
            target_label=target_label,
        )
        non_noise_label, non_noise_confidence = best_non_noise_prediction(selected_probs, self.labels)

        payload = {
            "prediction": final_label,
            "confidence": float(final_confidence),
            "raw_prediction": raw_label,
            "raw_confidence": raw_confidence,
            "best_non_noise": non_noise_label,
            "best_non_noise_confidence": float(non_noise_confidence),
            "decision": decision_detail,
            "decision_mode": self.config.decision_mode,
            "top": top_entries(selected_probs, self.labels, self.config.top_k),
            "scan": {
                "raw_iq_samples": int(raw_iq.shape[0]),
                "window_samples": int(self.config.window_samples),
                "candidate_starts": [int(start) for start in starts],
                "selected_start": selected_start,
                "selected_index": int(best_idx),
                "resolved_window_score_mode": score_mode,
                "candidates": candidates,
            },
            "capture_stats": capture_stats(selected_iq),
            "input_shape": list(model_windows.shape),
        }
        if target_label:
            payload["target_class"] = target_label
            payload["correct"] = final_label == target_label
            if target_label in self.labels:
                target_idx = self.labels.index(target_label)
                payload["target_conditional_confidence"] = conditional_class_confidence(
                    selected_probs,
                    self.labels,
                    target_idx,
                )
        return payload
