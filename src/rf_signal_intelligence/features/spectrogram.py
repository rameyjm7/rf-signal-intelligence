"""Spectrogram utilities extracted from the NoisyDroneRFv2 notebooks."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpectrogramConfig:
    """Full-complex STFT feature parameters."""

    sample_len: int = 1_048_576
    nfft: int = 1024
    hop: int = 1024
    time_bins: int = 1024
    burst_smooth_samples: int = 512
    clip_value: float = 6.0

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.nfft, self.time_bins, 2)


def find_burst_start(iq: np.ndarray, window_len: int, *, smooth_samples: int = 512) -> int:
    """Return a start index centered on the highest smoothed IQ power region."""
    if iq.shape[0] <= window_len:
        return 0
    power = np.mean(np.square(iq.astype(np.float32)), axis=1)
    smooth_len = max(1, min(smooth_samples, power.shape[0] // 8))
    if smooth_len > 1:
        kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
        power = np.convolve(power, kernel, mode="same")
    center = int(np.argmax(power))
    return int(np.clip(center - window_len // 2, 0, iq.shape[0] - window_len))


def normalize_iq_window(iq: np.ndarray) -> np.ndarray:
    """Center and RMS-normalize an IQ window."""
    window = np.asarray(iq[:, :2], dtype=np.float32)
    window = window - np.mean(window, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(window)) + 1e-8)
    return window / scale


def normalize_iq_per_sample(x: np.ndarray) -> np.ndarray:
    """Normalize each IQ sample by its maximum absolute value."""
    values = np.asarray(x, dtype=np.float32)
    normalized = np.empty_like(values)
    for idx in range(values.shape[0]):
        scale = np.max(np.abs(values[idx])) + 1e-12
        normalized[idx] = values[idx] / scale
    return normalized


def append_snr_feature(x: np.ndarray, snr: np.ndarray, *, scale: float = 20.0) -> np.ndarray:
    """Append normalized SNR as a third feature channel."""
    values = np.asarray(x, dtype=np.float32)
    snr_values = np.asarray(snr, dtype=np.float32)
    rows = []
    for idx in range(values.shape[0]):
        snr_col = np.full((values.shape[1], 1), snr_values[idx] / scale, dtype=np.float32)
        rows.append(np.concatenate([values[idx], snr_col], axis=1))
    return np.asarray(rows, dtype=np.float32)


def resize_time_axis(arr: np.ndarray, target_bins: int) -> np.ndarray:
    """Resize a spectrogram time axis using row-wise linear interpolation."""
    if arr.shape[1] == target_bins:
        return arr
    src_x = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    return np.stack([np.interp(dst_x, src_x, row).astype(np.float32) for row in arr], axis=0)


def iq_to_full_complex_spectrogram(
    iq: np.ndarray,
    config: SpectrogramConfig | None = None,
) -> np.ndarray:
    """Convert IQ samples into a full-complex STFT tensor `(freq, time, real/imag)`."""
    cfg = config or SpectrogramConfig()
    window_iq = normalize_iq_window(iq)
    complex_iq = window_iq[:, 0].astype(np.float32) + 1j * window_iq[:, 1].astype(np.float32)
    if len(complex_iq) < cfg.nfft:
        complex_iq = np.pad(complex_iq, (0, cfg.nfft - len(complex_iq)), mode="constant")
    starts = np.arange(0, len(complex_iq) - cfg.nfft + 1, cfg.hop)
    if starts.size == 0:
        starts = np.array([0])
    fft_window = np.hanning(cfg.nfft).astype(np.float32)
    frames = np.stack([complex_iq[s : s + cfg.nfft] * fft_window for s in starts], axis=0)
    fft_complex = np.fft.fftshift(np.fft.fft(frames, n=cfg.nfft, axis=1), axes=1).T
    fft_complex = fft_complex / float(cfg.nfft)
    real_part = resize_time_axis(fft_complex.real.astype(np.float32), cfg.time_bins)
    imag_part = resize_time_axis(fft_complex.imag.astype(np.float32), cfg.time_bins)
    spec = np.stack([real_part, imag_part], axis=-1).astype(np.float32)
    spec = spec / (np.std(spec) + 1e-6)
    return np.clip(spec, -cfg.clip_value, cfg.clip_value).astype(np.float32)


def safe_load_spectrogram_cache(cache_path: str | Path, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    """Load a cached spectrogram, returning None and removing corrupt cache files."""
    path = Path(cache_path)
    try:
        with np.load(path) as data:
            value = data["x"].astype(np.float32)
        if value.shape != expected_shape:
            raise ValueError(f"cached shape {value.shape} != expected {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError("cached spectrogram contains NaN or Inf")
        return value
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return None


def write_spectrogram_cache_atomic(cache_path: str | Path, value: np.ndarray) -> None:
    """Write an `.npz` spectrogram cache atomically."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, x=value.astype(np.float32))
    tmp_path.replace(path)
