#!/usr/bin/env python3
"""Listen for RF energy and classify it with the Noisy Drone RF v2 model.

Examples:
  python scripts/live_noisy_drone_rf_classifier.py --freq 2.44e9 --sample-rate 20e6
  python scripts/live_noisy_drone_rf_classifier.py --iq-file capture.npy --once
  python scripts/live_noisy_drone_rf_classifier.py --tx --freq 2.44e9 --sample-rate 20e6 --once

Live SDR capture uses SoapySDR when available. File replay accepts .npy, .npz,
.pt, and raw complex64 .bin/.c64 files.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import datetime as dt
import os
import pickle
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "noisy_drone_rf_v2"
    / "noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras"
)
LABEL_NAMES = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Noise", "Taranis", "Turnigy"]
DEFAULT_RX_DEVICE_ARGS = "driver=hackrf"
DEFAULT_TX_DEVICE_ARGS = "driver=bladerf,serial=7faa712b1fab42f4b84e494171b91721"
DEFAULT_BLADERF_STREAM_ARGS = "buffers=16,buflen=65536,transfers=8"
DEFAULT_FREQ = 2.399e9
DEFAULT_SAMPLE_RATE = 20e6
DEFAULT_BANDWIDTH = 20e6
DEFAULT_TX_DATASET_DIR = Path("/data/rameyjm7/datasets/NoisyDroneRFv2")
DEFAULT_TX_CLASS_NAME = "FutabaT14"
DEFAULT_TX_MIN_SNR = 20
NOISY_DRONE_SAMPLE_RE = re.compile(
    r"IQdata_sample(?P<sample>\d+)_target(?P<target>-?\d+)_snr(?P<snr>-?\d+)\.pt$"
)


class TorchStorageRef:
    def __init__(self, archive: zipfile.ZipFile, prefix: str, storage_type: str, key: str) -> None:
        self.archive = archive
        self.prefix = prefix
        self.storage_type = storage_type
        self.key = key

    @property
    def dtype(self) -> np.dtype:
        dtype_by_storage = {
            "FloatStorage": np.dtype("<f4"),
            "DoubleStorage": np.dtype("<f8"),
            "HalfStorage": np.dtype("<f2"),
            "LongStorage": np.dtype("<i8"),
            "IntStorage": np.dtype("<i4"),
            "ShortStorage": np.dtype("<i2"),
            "ByteStorage": np.dtype("u1"),
            "CharStorage": np.dtype("i1"),
            "BoolStorage": np.dtype("?"),
        }
        storage_name = self.storage_type.rsplit(".", 1)[-1]
        if storage_name not in dtype_by_storage:
            raise TypeError(f"Unsupported torch storage type in .pt fallback: {self.storage_type}")
        return dtype_by_storage[storage_name]

    def array(self) -> np.ndarray:
        payload = self.archive.read(f"{self.prefix}/data/{self.key}")
        return np.frombuffer(payload, dtype=self.dtype)


class TorchPtFallbackUnpickler(pickle.Unpickler):
    def __init__(self, *args, archive: zipfile.ZipFile, prefix: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.archive = archive
        self.prefix = prefix

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id in .pt fallback: {pid!r}")
        _, storage_type, key, _location, _size = pid[:5]
        storage_name = getattr(storage_type, "__name__", str(storage_type))
        return TorchStorageRef(self.archive, self.prefix, storage_name, str(key))

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_torch_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict

            return OrderedDict
        return super().find_class(module, name)


def rebuild_torch_tensor_v2(storage, storage_offset, size, stride, _requires_grad, _backward_hooks):
    if not isinstance(storage, TorchStorageRef):
        raise TypeError(f"Expected TorchStorageRef, got {type(storage).__name__}")
    base = storage.array()
    shape = tuple(int(dim) for dim in size)
    strides = tuple(int(item) * base.dtype.itemsize for item in stride)
    offset = int(storage_offset) * base.dtype.itemsize
    tensor = np.lib.stride_tricks.as_strided(base[offset // base.dtype.itemsize :], shape=shape, strides=strides)
    return np.array(tensor, copy=True)


def power_db(iq: np.ndarray) -> float:
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    return float(10.0 * np.log10(np.mean(np.abs(complex_iq) ** 2) + 1e-12))


def normalize_iq_window(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq[:, :2], dtype=np.float32)
    iq = iq - np.mean(iq, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(iq)) + 1e-8)
    return iq / scale


def resize_time_axis(arr: np.ndarray, target_bins: int) -> np.ndarray:
    if arr.shape[1] == target_bins:
        return arr
    src_x = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    return np.stack([np.interp(dst_x, src_x, row).astype(np.float32) for row in arr], axis=0)


def iq_window_to_spectrogram(
    iq: np.ndarray,
    *,
    nfft: int,
    hop: int,
    time_bins: int,
) -> np.ndarray:
    iq = normalize_iq_window(iq[:, :2])
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    if len(complex_iq) < nfft:
        complex_iq = np.pad(complex_iq, (0, nfft - len(complex_iq)), mode="constant")

    starts = np.arange(0, len(complex_iq) - nfft + 1, hop)
    if starts.size == 0:
        starts = np.array([0])

    window = np.hanning(nfft).astype(np.float32)
    frames = np.stack([complex_iq[start : start + nfft] * window for start in starts], axis=0)
    fft_complex = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1).T
    fft_complex = fft_complex / float(nfft)

    real_part = resize_time_axis(fft_complex.real.astype(np.float32), time_bins)
    imag_part = resize_time_axis(fft_complex.imag.astype(np.float32), time_bins)
    spec = np.stack([real_part, imag_part], axis=-1).astype(np.float32)
    spec = spec / (np.std(spec) + 1e-6)
    return np.clip(spec, -6.0, 6.0).astype(np.float32)


def coerce_iq_array(value: object) -> np.ndarray:
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if np.iscomplexobj(arr):
            arr = np.stack([arr.real, arr.imag], axis=-1)
        else:
            if arr.size % 2 != 0:
                arr = arr[:-1]
            arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        elif arr.shape[-1] != 2:
            raise ValueError(f"Expected IQ with final dimension 2, got {arr.shape}")
    else:
        if arr.shape[-1] == 2:
            arr = arr.reshape(-1, 2)
        elif arr.shape[0] == 2:
            arr = np.moveaxis(arr, 0, -1).reshape(-1, 2)
        else:
            raise ValueError(f"Expected IQ tensor with two channels, got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def load_iq_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return coerce_iq_array(np.load(path))
    if suffix == ".npz":
        with np.load(path) as data:
            key = "iq" if "iq" in data else "x" if "x" in data else data.files[0]
            return coerce_iq_array(data[key])
    if suffix == ".pt":
        try:
            import torch
        except ImportError:
            obj = load_torch_pt_without_torch(path)
        else:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("x_iq", "iq", "IQ", "x", "X", "data", "samples", "signal", "waveform"):
                if key in obj:
                    return coerce_iq_array(obj[key])
        return coerce_iq_array(obj[0] if isinstance(obj, (list, tuple)) else obj)
    if suffix in {".bin", ".c64", ".complex64"}:
        return coerce_iq_array(np.fromfile(path, dtype=np.complex64))
    raise ValueError(f"Unsupported IQ file type: {path}")


def load_torch_pt_without_torch(path: Path):
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            f"{path} is a legacy/non-zip PyTorch file. Install torch or convert it to .npy/.npz first."
        )
    with zipfile.ZipFile(path) as archive:
        pkl_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if not pkl_names:
            raise RuntimeError(f"{path} does not look like a tensor .pt archive.")
        pkl_name = pkl_names[0]
        prefix = pkl_name.rsplit("/", 1)[0]
        return TorchPtFallbackUnpickler(
            archive.open(pkl_name),
            archive=archive,
            prefix=prefix,
        ).load()


class FileIqSource:
    def __init__(self, iq: np.ndarray, *, loop: bool) -> None:
        self.iq = iq
        self.loop = loop
        self.offset = 0

    def read_iq(self, count: int) -> np.ndarray:
        if self.offset >= len(self.iq):
            if not self.loop:
                return np.empty((0, 2), dtype=np.float32)
            self.offset = 0
        end = min(self.offset + count, len(self.iq))
        chunk = self.iq[self.offset:end]
        self.offset = end
        return chunk.astype(np.float32, copy=False)

    def close(self) -> None:
        return None


class SoapyIqSource:
    def __init__(
        self,
        *,
        device_args: str,
        channel: int,
        sample_rate: float,
        freq: float,
        gain: float | None,
        agc: bool,
        antenna: str | None,
        bandwidth: float | None,
        stream_args: str | None,
    ) -> None:
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX
        except ImportError as exc:
            raise RuntimeError(
                "Live SDR capture requires SoapySDR Python bindings. "
                "Use --iq-file for replay if no SDR stack is installed."
            ) from exc

        self.SoapySDR = SoapySDR
        self.rx = SOAPY_SDR_RX
        self.device = make_soapy_device(SoapySDR, device_args, role="RX")
        self.channel = channel
        self.device.setSampleRate(SOAPY_SDR_RX, channel, sample_rate)
        self.device.setFrequency(SOAPY_SDR_RX, channel, freq)
        if antenna is not None:
            antennas = list(self.device.listAntennas(SOAPY_SDR_RX, channel))
            if antenna in antennas:
                self.device.setAntenna(SOAPY_SDR_RX, channel, antenna)
            elif antennas:
                print(
                    f"Requested antenna {antenna!r} is not available on RX channel {channel}; "
                    f"available antennas: {', '.join(antennas)}",
                    file=sys.stderr,
                )
        if bandwidth is not None:
            self.device.setBandwidth(SOAPY_SDR_RX, channel, bandwidth)
        if hasattr(self.device, "hasGainMode") and self.device.hasGainMode(SOAPY_SDR_RX, channel):
            self.device.setGainMode(SOAPY_SDR_RX, channel, bool(agc))
        if gain is not None:
            self.device.setGain(SOAPY_SDR_RX, channel, gain)
        self.gain_mode = (
            self.device.getGainMode(SOAPY_SDR_RX, channel)
            if hasattr(self.device, "hasGainMode") and self.device.hasGainMode(SOAPY_SDR_RX, channel)
            else None
        )
        self.actual_gain = self.device.getGain(SOAPY_SDR_RX, channel) if gain is not None else None
        self.overflow_count = 0
        self.stream = self.device.setupStream(
            SOAPY_SDR_RX,
            SOAPY_SDR_CF32,
            [channel],
            resolve_stream_args(device_args, stream_args),
        )
        self.device.activateStream(self.stream)

    def read_iq(self, count: int) -> np.ndarray:
        buff = np.empty(count, dtype=np.complex64)
        filled = 0
        while filled < count:
            view = buff[filled:]
            result = self.device.readStream(self.stream, [view], len(view), timeoutUs=1_000_000)
            if result.ret > 0:
                filled += result.ret
            elif result.ret == self.SoapySDR.SOAPY_SDR_TIMEOUT:
                if filled:
                    break
            elif result.ret == self.SoapySDR.SOAPY_SDR_OVERFLOW:
                self.overflow_count += 1
                if self.overflow_count <= 5 or self.overflow_count % 25 == 0:
                    print(f"RX overflow ({self.overflow_count}); continuing", file=sys.stderr, flush=True)
                if filled:
                    break
            elif result.ret == self.SoapySDR.SOAPY_SDR_STREAM_ERROR:
                raise RuntimeError(
                    "SoapySDR RX stream error. If bladeRF logged LIBUSB_ERROR_NO_MEM, reduce "
                    "--rx-stream-args/--tx-stream-args or --sample-rate. Try "
                    "--rx-stream-args 'buffers=8,buflen=32768,transfers=4'."
                )
            else:
                raise RuntimeError(f"SoapySDR readStream failed with code {result.ret}")
        return coerce_iq_array(buff[:filled])

    def close(self) -> None:
        self.device.deactivateStream(self.stream)
        self.device.closeStream(self.stream)


class SoapyIqSink:
    def __init__(
        self,
        *,
        device_args: str,
        channel: int,
        sample_rate: float,
        freq: float,
        gain: float | None,
        antenna: str | None,
        bandwidth: float | None,
        stream_args: str | None,
    ) -> None:
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_TX
        except ImportError as exc:
            raise RuntimeError("TX requires SoapySDR Python bindings.") from exc

        self.SoapySDR = SoapySDR
        self.tx = SOAPY_SDR_TX
        self.device = make_soapy_device(SoapySDR, device_args, role="TX")
        self.channel = channel
        self.device.setSampleRate(SOAPY_SDR_TX, channel, sample_rate)
        self.device.setFrequency(SOAPY_SDR_TX, channel, freq)
        if antenna is not None:
            antennas = list(self.device.listAntennas(SOAPY_SDR_TX, channel))
            if antenna in antennas:
                self.device.setAntenna(SOAPY_SDR_TX, channel, antenna)
            elif antennas:
                print(
                    f"Requested TX antenna {antenna!r} is not available on TX channel {channel}; "
                    f"available antennas: {', '.join(antennas)}",
                    file=sys.stderr,
                )
        if bandwidth is not None:
            self.device.setBandwidth(SOAPY_SDR_TX, channel, bandwidth)
        if gain is not None:
            self.device.setGain(SOAPY_SDR_TX, channel, gain)
        self.stream = self.device.setupStream(
            SOAPY_SDR_TX,
            SOAPY_SDR_CF32,
            [channel],
            resolve_stream_args(device_args, stream_args),
        )
        self.device.activateStream(self.stream)

    def write_iq(self, iq: np.ndarray, *, chunk_samples: int) -> None:
        complex_iq = iq_to_complex64(iq)
        offset = 0
        while offset < len(complex_iq):
            chunk = complex_iq[offset : offset + chunk_samples]
            result = self.device.writeStream(
                self.stream,
                [chunk],
                len(chunk),
                timeoutUs=1_000_000,
            )
            if result.ret > 0:
                offset += result.ret
            else:
                raise RuntimeError(f"SoapySDR writeStream failed with code {result.ret}")

    def close(self) -> None:
        self.device.deactivateStream(self.stream)
        self.device.closeStream(self.stream)


class TxWorker:
    def __init__(self, sink: SoapyIqSink, iq: np.ndarray, *, chunk_samples: int, repeats: int) -> None:
        self.sink = sink
        self.iq = iq
        self.chunk_samples = chunk_samples
        self.repeats = repeats
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.error = None
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="noisy-drone-tx", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            count = 0
            while not self.stop_event.is_set() and (self.repeats <= 0 or count < self.repeats):
                self.sink.write_iq(self.iq, chunk_samples=self.chunk_samples)
                count += 1
        except BaseException as exc:
            self.error = exc

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        if self.error is not None:
            raise RuntimeError(f"TX failed: {self.error}") from self.error


def calibrate_noise_floor(source: FileIqSource | SoapyIqSource, *, chunk_samples: int, chunks: int) -> float:
    readings = []
    for _ in range(chunks):
        chunk = source.read_iq(chunk_samples)
        if len(chunk) == 0:
            break
        readings.append(power_db(chunk))
    if not readings:
        raise RuntimeError("Could not read any samples during noise-floor calibration.")
    return float(np.median(readings))


def parse_key_value_args(value: str) -> dict[str, str]:
    args = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            args[item] = ""
            continue
        key, arg_value = item.split("=", 1)
        args[key.strip()] = arg_value.strip()
    return args


def make_soapy_device(soapy_module, device_args: str, *, role: str):
    parsed_args = parse_key_value_args(device_args)
    try:
        return soapy_module.Device(parsed_args)
    except RuntimeError as exc:
        message = str(exc)
        if "getHardwareTime() -19" in message or "Insufficient initialization" in message:
            raise RuntimeError(
                f"Could not open {role} device {device_args!r}: the bladeRF is not initialized. "
                "Install/load the matching FPGA image first, for example `bladerf-fpga-hostedxa4` "
                "for bladeRF 2.0 Micro xA4 or `bladerf-fpga-hostedxa9` for xA9, then load or "
                "flash it with bladeRF-cli."
            ) from exc
        if parsed_args.get("driver") == "hackrf":
            raise RuntimeError(
                f"Could not open {role} HackRF device {device_args!r}. "
                "`hackrf_info` must show the board first; check USB, permissions, and that no other "
                "program has it open."
            ) from exc
        raise RuntimeError(f"Could not open {role} SDR device {device_args!r}: {message}") from exc


def frontend_label(device_args: str, role: str, channel: int) -> str:
    parsed_args = parse_key_value_args(device_args)
    if parsed_args.get("driver", "").lower() == "bladerf":
        if role.upper() == "RX":
            return f"bladeRF RX{channel + 1}"
        if role.upper() == "TX":
            return f"bladeRF TX{channel + 1}"
    return f"{role.upper()} channel {channel}"


def resolve_stream_args(device_args: str, stream_args: str | None) -> dict[str, str]:
    if stream_args is not None:
        return parse_key_value_args(stream_args)
    parsed_device_args = parse_key_value_args(device_args)
    if parsed_device_args.get("driver", "").lower() == "bladerf":
        return parse_key_value_args(DEFAULT_BLADERF_STREAM_ARGS)
    return {}


def iq_to_complex64(iq: np.ndarray) -> np.ndarray:
    iq = coerce_iq_array(iq)
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    return complex_iq.astype(np.complex64, copy=False)


def prepare_tx_iq(iq: np.ndarray, *, amplitude: float, pad_seconds: float, sample_rate: float) -> np.ndarray:
    complex_iq = iq_to_complex64(iq)
    complex_iq = complex_iq - np.mean(complex_iq)
    peak = float(np.max(np.abs(complex_iq))) if complex_iq.size else 0.0
    if peak > 0.0:
        complex_iq = complex_iq / peak
    complex_iq = (complex_iq * float(amplitude)).astype(np.complex64)
    pad_samples = max(0, int(round(pad_seconds * sample_rate)))
    if pad_samples:
        silence = np.zeros(pad_samples, dtype=np.complex64)
        complex_iq = np.concatenate([silence, complex_iq, silence])
    return coerce_iq_array(complex_iq)


def discover_noisy_drone_dataset(search_roots: list[Path]) -> Path | None:
    preferred_names = ("noisy_drone_rx", "NoisyDroneRFv2", "noisy_drone_rf_v2")
    for root in search_roots:
        if not root.exists():
            continue
        for preferred in preferred_names:
            for path in root.rglob(preferred):
                if path.is_dir() and any(path.rglob("IQdata_sample*_target*_snr*.pt")):
                    return path
        for class_stats in root.rglob("class_stats.csv"):
            parent = class_stats.parent
            if any(parent.rglob("IQdata_sample*_target*_snr*.pt")):
                return parent
    return None


def noisy_drone_file_metadata(path: Path) -> tuple[int | None, int | None]:
    match = NOISY_DRONE_SAMPLE_RE.search(path.name)
    if not match:
        return None, None
    return int(match.group("target")), int(match.group("snr"))


def choose_tx_sample(
    dataset_dir: Path | None,
    tx_iq_file: Path | None,
    *,
    seed: int | None,
    min_snr: int | None,
    target: int | None,
    class_name: str | None,
) -> tuple[Path, np.ndarray]:
    if tx_iq_file is not None:
        return tx_iq_file, load_iq_file(tx_iq_file)

    search_roots = [Path("/data"), Path("/scratch")]
    dataset_dir = dataset_dir or discover_noisy_drone_dataset(search_roots)
    if dataset_dir is None:
        raise FileNotFoundError(
            "Could not find noisy_drone_rx/NoisyDroneRFv2 samples under /data or /scratch. "
            "Pass --tx-dataset-dir or --tx-iq-file explicitly."
        )

    files = sorted(dataset_dir.rglob("IQdata_sample*_target*_snr*.pt"))
    if not files:
        raise FileNotFoundError(f"No IQdata_sample*_target*_snr*.pt files found under {dataset_dir}")

    if class_name is not None:
        normalized = class_name.casefold()
        matching_targets = [idx for idx, name in enumerate(LABEL_NAMES) if name.casefold() == normalized]
        if not matching_targets:
            raise ValueError(f"Unknown class name {class_name!r}; choose one of {', '.join(LABEL_NAMES)}")
        target = matching_targets[0]

    filtered = []
    for path in files:
        file_target, file_snr = noisy_drone_file_metadata(path)
        if target is not None and file_target != target:
            continue
        if min_snr is not None and (file_snr is None or file_snr < min_snr):
            continue
        filtered.append(path)
    files = filtered
    if not files:
        raise FileNotFoundError(
            f"No dataset samples matched target={target} class_name={class_name!r} min_snr={min_snr}"
        )

    rng = random.Random(seed)
    path = rng.choice(files)
    return path, load_iq_file(path)


def describe_noisy_drone_sample(path: Path) -> str:
    match = NOISY_DRONE_SAMPLE_RE.search(path.name)
    if not match:
        return path.name
    target = int(match.group("target"))
    label = LABEL_NAMES[target] if 0 <= target < len(LABEL_NAMES) else f"target_{target}"
    return f"{path.name} true={label} target={target} snr={match.group('snr')}dB"


def capture_on_energy(
    source: FileIqSource | SoapyIqSource,
    *,
    chunk_samples: int,
    window_samples: int,
    threshold_db: float,
    calibration_chunks: int,
    absolute_threshold_db: float | None,
    pretrigger_chunks: int,
    on_calibrated=None,
    trigger_timeout_sec: float | None = None,
) -> tuple[np.ndarray | None, float, float]:
    floor_db = calibrate_noise_floor(source, chunk_samples=chunk_samples, chunks=calibration_chunks)
    trigger_db = absolute_threshold_db if absolute_threshold_db is not None else floor_db + threshold_db
    print(f"Noise floor {floor_db:.1f} dB, trigger {trigger_db:.1f} dB", flush=True)
    if on_calibrated is not None:
        on_calibrated(floor_db, trigger_db)

    history: deque[np.ndarray] = deque(maxlen=max(0, pretrigger_chunks))
    wait_started = time.monotonic()
    while True:
        if trigger_timeout_sec is not None and time.monotonic() - wait_started > trigger_timeout_sec:
            return None, floor_db, trigger_db
        chunk = source.read_iq(chunk_samples)
        if len(chunk) == 0:
            return None, floor_db, trigger_db
        level_db = power_db(chunk)
        if level_db < trigger_db:
            history.append(chunk)
            continue

        print(f"Signal detected at {level_db:.1f} dB; capturing {window_samples} samples", flush=True)
        chunks = list(history)
        chunks.append(chunk)
        total = sum(len(item) for item in chunks)
        while total < window_samples:
            next_chunk = source.read_iq(min(chunk_samples, window_samples - total))
            if len(next_chunk) == 0:
                break
            chunks.append(next_chunk)
            total += len(next_chunk)
        capture = np.concatenate(chunks, axis=0)[:window_samples]
        if len(capture) < window_samples:
            capture = np.pad(capture, ((0, window_samples - len(capture)), (0, 0)), mode="constant")
        return capture.astype(np.float32, copy=False), floor_db, trigger_db


def flush_rx(source: FileIqSource | SoapyIqSource, *, chunk_samples: int, chunks: int) -> None:
    for _ in range(max(0, chunks)):
        chunk = source.read_iq(chunk_samples)
        if len(chunk) == 0:
            return


def capture_fixed_window(
    source: FileIqSource | SoapyIqSource,
    *,
    chunk_samples: int,
    capture_samples: int,
) -> np.ndarray | None:
    chunks = []
    total = 0
    while total < capture_samples:
        chunk = source.read_iq(min(chunk_samples, capture_samples - total))
        if len(chunk) == 0:
            break
        chunks.append(chunk)
        total += len(chunk)
    if not chunks:
        return None
    capture = np.concatenate(chunks, axis=0)[:capture_samples]
    if len(capture) < capture_samples:
        capture = np.pad(capture, ((0, capture_samples - len(capture)), (0, 0)), mode="constant")
    return capture.astype(np.float32, copy=False)


def select_high_power_window(iq: np.ndarray, *, window_samples: int, smooth_samples: int) -> tuple[np.ndarray, int]:
    iq = coerce_iq_array(iq)
    if len(iq) <= window_samples:
        if len(iq) < window_samples:
            iq = np.pad(iq, ((0, window_samples - len(iq)), (0, 0)), mode="constant")
        return iq.astype(np.float32, copy=False), 0

    power = np.mean(np.square(iq.astype(np.float32)), axis=1)
    smooth_len = max(1, min(int(smooth_samples), power.shape[0] // 8))
    if smooth_len > 1:
        kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
        power = np.convolve(power, kernel, mode="same")
    center = int(np.argmax(power))
    start = int(np.clip(center - window_samples // 2, 0, len(iq) - window_samples))
    return iq[start : start + window_samples].astype(np.float32, copy=False), start


def candidate_window_starts(total_samples: int, window_samples: int, stride_samples: int) -> list[int]:
    if total_samples <= window_samples:
        return [0]
    stride_samples = max(1, int(stride_samples))
    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    final_start = total_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def classify_iq(
    model,
    iq: np.ndarray,
    *,
    nfft: int,
    hop: int,
    time_bins: int,
    labels: list[str],
    phase_tta: int = 1,
) -> tuple[str, float, np.ndarray]:
    phase_tta = max(1, int(phase_tta))
    if phase_tta == 1:
        spec = iq_window_to_spectrogram(iq, nfft=nfft, hop=hop, time_bins=time_bins)
        probs = model.predict(spec[None, ...], verbose=0)[0].astype(np.float64)
    else:
        complex_iq = iq_to_complex64(iq)
        specs = []
        for phase in np.linspace(0.0, 2.0 * np.pi, phase_tta, endpoint=False):
            rotated = complex_iq * np.exp(1j * phase)
            specs.append(iq_window_to_spectrogram(coerce_iq_array(rotated), nfft=nfft, hop=hop, time_bins=time_bins))
        probs = model.predict(np.stack(specs, axis=0), verbose=0).astype(np.float64).mean(axis=0)
    top_idx = int(np.argmax(probs))
    return labels[top_idx], float(probs[top_idx]), probs


def noise_class_index(labels: list[str]) -> int | None:
    return labels.index("Noise") if "Noise" in labels else None


def non_noise_probability_mass(probs: np.ndarray, labels: list[str]) -> float:
    noise_idx = noise_class_index(labels)
    return float(np.sum(probs) - (probs[noise_idx] if noise_idx is not None else 0.0))


def conditional_class_confidence(probs: np.ndarray, labels: list[str], class_idx: int) -> float:
    denominator = non_noise_probability_mass(probs, labels)
    return float(probs[class_idx] / denominator) if denominator > 0.0 else float(probs[class_idx])


def best_non_noise_prediction(probs: np.ndarray, labels: list[str]) -> tuple[str, float]:
    noise_idx = labels.index("Noise") if "Noise" in labels else None
    masked = probs.astype(np.float64, copy=True)
    if noise_idx is not None:
        masked[noise_idx] = -np.inf
    top_idx = int(np.argmax(masked))
    confidence = conditional_class_confidence(probs, labels, top_idx)
    return labels[top_idx], confidence


def choose_final_prediction(
    raw_label: str,
    raw_confidence: float,
    probs: np.ndarray,
    labels: list[str],
    *,
    decision_mode: str,
    non_noise_threshold: float,
    signal_present: bool,
    target_label: str | None,
) -> tuple[str, float, str | None]:
    if decision_mode == "raw":
        return raw_label, raw_confidence, None

    non_noise_label, non_noise_confidence = best_non_noise_prediction(probs, labels)
    if decision_mode == "non-noise":
        return non_noise_label, non_noise_confidence, "decision=non-noise"

    if raw_label == "Noise" and signal_present and target_label in labels and target_label != "Noise":
        target_idx = labels.index(target_label)
        target_confidence = conditional_class_confidence(probs, labels, target_idx)
        if target_confidence >= non_noise_threshold:
            return (
                target_label,
                target_confidence,
                f"decision=hybrid promoted_target threshold={non_noise_threshold:.3f}",
            )
        return raw_label, raw_confidence, (
            f"decision=hybrid kept_noise target={target_label}:{target_confidence:.3f} "
            f"threshold={non_noise_threshold:.3f}"
        )

    if raw_label == "Noise" and signal_present and non_noise_confidence >= non_noise_threshold:
        return (
            non_noise_label,
            non_noise_confidence,
            f"decision=hybrid promoted_non_noise threshold={non_noise_threshold:.3f}",
        )
    return raw_label, raw_confidence, None


def print_prediction(label: str, confidence: float, probs: np.ndarray, labels: list[str], *, top_k: int) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    ranking = np.argsort(probs)[::-1][:top_k]
    top = ", ".join(f"{labels[idx]}={probs[idx]:.3f}" for idx in ranking)
    print(f"{now} prediction={label} confidence={confidence:.3f} top={top}", flush=True)


def capture_stats(iq: np.ndarray) -> dict[str, float]:
    iq = coerce_iq_array(iq)
    complex_iq = iq_to_complex64(iq)
    magnitude = np.abs(complex_iq)
    return {
        "power_db": power_db(iq),
        "peak": float(np.max(magnitude)) if magnitude.size else 0.0,
        "i_clip_pct": float(np.mean(np.abs(iq[:, 0]) >= 0.99) * 100.0),
        "q_clip_pct": float(np.mean(np.abs(iq[:, 1]) >= 0.99) * 100.0),
        "mag_fullscale_pct": float(np.mean(magnitude >= 1.0) * 100.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--iq-file", type=Path, help="Replay IQ from a file instead of live SDR.")
    parser.add_argument("--loop-file", action="store_true", help="Loop --iq-file after EOF.")
    parser.add_argument("--once", action="store_true", help="Exit after the first classification.")
    parser.add_argument(
        "--device-args",
        "--rx-device-args",
        dest="rx_device_args",
        default=DEFAULT_RX_DEVICE_ARGS,
        help="SoapySDR RX device args.",
    )
    parser.add_argument("--channel", "--rx-channel", dest="rx_channel", type=int, default=0, help="Soapy RX channel. For bladeRF, RX1 is channel 0.")
    parser.add_argument("--antenna", "--rx-antenna", dest="rx_antenna", default=None, help="Soapy RX antenna name to select when available.")
    parser.add_argument("--freq", type=float, default=DEFAULT_FREQ, help="Tuned RF center frequency in Hz for live SDR.")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH, help="Analog/filter bandwidth in Hz.")
    parser.add_argument("--gain", "--rx-gain", dest="rx_gain", type=float, default=60.0)
    parser.add_argument("--rx-agc", action="store_true", help="Enable RX AGC. By default AGC is disabled for manual gain control.")
    parser.add_argument(
        "--rx-stream-args",
        help=f"Soapy RX stream args. For bladeRF, defaults to {DEFAULT_BLADERF_STREAM_ARGS!r}.",
    )
    parser.add_argument("--window-samples", type=int, default=int(os.getenv("NOISY_DRONE_MAX_IQ_SAMPLES", "1048576")))
    parser.add_argument("--capture-samples", type=int, help="Raw RX samples to capture before selecting a model window.")
    parser.add_argument("--burst-smooth-samples", type=int, default=512, help="Smoothing length for selecting the highest-power RX window.")
    parser.add_argument("--scan-windows", action="store_true", default=True, help="Classify candidate RX windows and choose the best one.")
    parser.add_argument("--no-scan-windows", action="store_false", dest="scan_windows", help="Disable candidate-window scanning.")
    parser.add_argument("--scan-stride-samples", type=int, default=262144, help="Stride between candidate RX windows.")
    parser.add_argument(
        "--window-score-mode",
        choices=("auto", "target", "non-noise", "raw"),
        default="auto",
        help="How to choose the best candidate RX window.",
    )
    parser.add_argument("--chunk-samples", type=int, default=65536)
    parser.add_argument("--pretrigger-chunks", type=int, default=1)
    parser.add_argument("--calibration-chunks", type=int, default=20)
    parser.add_argument("--threshold-db", type=float, default=8.0)
    parser.add_argument("--absolute-threshold-db", type=float)
    parser.add_argument("--trigger-timeout-sec", type=float, default=None)
    parser.add_argument("--nfft", type=int, default=int(os.getenv("NOISY_DRONE_SPEC_NFFT", "1024")))
    parser.add_argument("--hop", type=int, default=int(os.getenv("NOISY_DRONE_SPEC_HOP", "1024")))
    parser.add_argument("--time-bins", type=int, default=int(os.getenv("NOISY_DRONE_SPEC_TIME_BINS", "1024")))
    parser.add_argument("--phase-tta", type=int, default=1, help="Average predictions over this many complex phase rotations.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--decision-mode",
        choices=("hybrid", "raw", "non-noise"),
        default="hybrid",
        help=(
            "Final label policy. hybrid keeps raw model labels except when Noise wins despite a known/captured "
            "signal and a strong conditional non-noise class."
        ),
    )
    parser.add_argument(
        "--non-noise-threshold",
        type=float,
        default=0.55,
        help="Conditional non-noise confidence needed for hybrid promotion from Noise.",
    )
    parser.add_argument(
        "--signal-margin-db",
        type=float,
        default=4.0,
        help="RX capture power above calibrated noise floor that counts as signal present outside TX mode.",
    )
    parser.add_argument(
        "--ignore-noise-class",
        action="store_true",
        help="Deprecated alias for --decision-mode non-noise.",
    )
    parser.add_argument("--cooldown-sec", type=float, default=1.0)
    parser.add_argument("--tx", action="store_true", help="Transmit a dataset IQ sample before receiving/classifying.")
    parser.add_argument("--tx-device-args", default=DEFAULT_TX_DEVICE_ARGS, help="SoapySDR TX device args.")
    parser.add_argument("--tx-channel", type=int, default=0, help="Soapy TX channel. For bladeRF, TX1 is channel 0.")
    parser.add_argument("--tx-antenna", default="TX", help="Soapy TX antenna name to select when available.")
    parser.add_argument("--tx-gain", type=float, default=60.0)
    parser.add_argument("--tx-bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument(
        "--tx-stream-args",
        help=f"Soapy TX stream args. For bladeRF, defaults to {DEFAULT_BLADERF_STREAM_ARGS!r}.",
    )
    parser.add_argument("--tx-iq-file", type=Path, help="Specific IQ file to transmit.")
    parser.add_argument("--tx-dataset-dir", type=Path, default=DEFAULT_TX_DATASET_DIR, help="Directory containing NoisyDroneRFv2 .pt samples.")
    parser.add_argument("--tx-min-snr", type=int, default=DEFAULT_TX_MIN_SNR, help="Only choose NoisyDroneRFv2 samples with this SNR or higher.")
    parser.add_argument("--tx-target", type=int, help="Only choose this NoisyDroneRFv2 target index.")
    parser.add_argument("--tx-class-name", choices=LABEL_NAMES, default=DEFAULT_TX_CLASS_NAME, help="Only choose this NoisyDroneRFv2 class.")
    parser.add_argument("--tx-amplitude", type=float, default=0.2)
    parser.add_argument("--tx-pad-sec", type=float, default=0.005)
    parser.add_argument("--tx-repeat", type=int, default=0, help="TX repeats. Use 0 to transmit until RX capture finishes.")
    parser.add_argument("--tx-seed", type=int)
    parser.add_argument("--tx-delay-sec", type=float, default=0.2)
    parser.add_argument(
        "--tx-capture-mode",
        choices=("direct", "energy"),
        default="direct",
        help="In TX mode, capture immediately after TX starts or wait for an energy trigger.",
    )
    parser.add_argument("--rx-flush-chunks", type=int, default=4, help="RX chunks to discard before TX capture.")
    parser.add_argument("--save-rx-iq", type=Path, help="Save the received/classified IQ window as .npy.")
    parser.add_argument("--tx-test-all-classes", action="store_true", help="Run one or more TX/RX trials for every requested class.")
    parser.add_argument(
        "--tx-test-classes",
        default=",".join(LABEL_NAMES),
        help="Comma-separated class names to sweep when --tx-test-all-classes is set.",
    )
    parser.add_argument("--tx-test-count", type=int, default=1, help="Trials per class for --tx-test-all-classes.")
    parser.add_argument(
        "--tx-test-output-csv",
        type=Path,
        default=Path("outputs/class_sweep.csv"),
        help="CSV summary path for --tx-test-all-classes.",
    )
    parser.add_argument(
        "--tx-test-output-md",
        type=Path,
        default=Path("results/noisy_drone_rf_v2/class_sweep_results.md"),
        help="Markdown report path for --tx-test-all-classes.",
    )
    parser.add_argument(
        "--tx-test-save-rx-dir",
        type=Path,
        default=Path("outputs/class_sweep_iq"),
        help="Directory for per-trial RX IQ windows.",
    )
    parser.add_argument(
        "--tx-test-save-plots-dir",
        type=Path,
        default=Path("outputs/class_sweep_plots"),
        help="Directory for per-trial waterfall PNG snapshots.",
    )
    parser.add_argument("--tx-test-seed-start", type=int, default=1000, help="Base TX seed for repeatable class sweep sample choices.")
    parser.add_argument("--stop-on-test-error", action="store_true", help="Stop the class sweep after the first failed child trial.")
    return parser.parse_args()


def add_cli_arg(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def parse_child_trial_log(lines: list[str]) -> dict[str, str]:
    row: dict[str, str] = {}
    patterns = {
        "tx_sample": re.compile(r"^TX sample: (?P<value>.+)$"),
        "tx_file": re.compile(
            r"^TX file model prediction=(?P<tx_file_prediction>\S+) "
            r"confidence=(?P<tx_file_confidence>[0-9.]+) "
            r"best_non_noise=(?P<tx_file_best_non_noise>\S+) "
            r"non_noise_confidence=(?P<tx_file_non_noise_confidence>[0-9.]+)"
        ),
        "selected_window": re.compile(
            r"^Selected .* window start=(?P<selected_window_start>\d+) "
            r"from raw_capture_samples=(?P<raw_capture_samples>\d+)"
        ),
        "capture_stats": re.compile(
            r"^Capture power (?P<capture_power_db>-?[0-9.]+) dB, peak (?P<capture_peak>[0-9.]+), "
            r"I clip (?P<i_clip_pct>[0-9.]+)%, Q clip (?P<q_clip_pct>[0-9.]+)%, "
            r"\|IQ\|>=1 (?P<mag_fullscale_pct>[0-9.]+)%"
        ),
        "best_non_noise": re.compile(
            r"^best_non_noise=(?P<best_non_noise>\S+) "
            r"non_noise_confidence=(?P<best_non_noise_confidence>[0-9.]+)"
        ),
        "target": re.compile(
            r"^target_class=(?P<target_class>\S+) "
            r"target_non_noise_confidence=(?P<target_non_noise_confidence>[0-9.]+)"
        ),
        "prediction": re.compile(
            r"^\S+ prediction=(?P<prediction>\S+) confidence=(?P<confidence>[0-9.]+) top=(?P<top>.+)$"
        ),
    }
    for line in lines:
        if line.startswith("decision="):
            row["decision"] = line
            continue
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if not match:
                continue
            if key == "tx_sample":
                row[key] = match.group("value")
            else:
                row.update(match.groupdict())
            break
    return row


def markdown_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def row_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_float(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def summarize_class_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row.get("target_class", "?"), []).append(row)

    summary = []
    for class_name, class_rows in by_class.items():
        total = len(class_rows)
        passed = sum(1 for row in class_rows if row.get("passed") == "1")
        target_confidences = [
            value for row in class_rows if (value := row_float(row, "target_non_noise_confidence")) is not None
        ]
        capture_powers = [value for row in class_rows if (value := row_float(row, "capture_power_db")) is not None]
        clip_rates = [value for row in class_rows if (value := row_float(row, "mag_fullscale_pct")) is not None]
        summary.append(
            {
                "class": class_name,
                "passed": str(passed),
                "total": str(total),
                "accuracy": format_float(passed / total if total else None),
                "min_target_confidence": format_float(min(target_confidences) if target_confidences else None),
                "mean_target_confidence": format_float(
                    sum(target_confidences) / len(target_confidences) if target_confidences else None
                ),
                "mean_capture_power_db": format_float(
                    sum(capture_powers) / len(capture_powers) if capture_powers else None,
                    digits=1,
                ),
                "max_fullscale_pct": format_float(max(clip_rates) if clip_rates else None),
            }
        )
    return summary


def confusion_matrix_counts(rows: list[dict[str, str]], classes: list[str]) -> dict[str, dict[str, int]]:
    labels = list(classes)
    for row in rows:
        prediction = row.get("prediction")
        if prediction and prediction not in labels:
            labels.append(prediction)
    matrix = {target: {prediction: 0 for prediction in labels} for target in labels}
    for row in rows:
        target = row.get("target_class", "?")
        prediction = row.get("prediction", "?")
        if target not in matrix:
            matrix[target] = {label: 0 for label in labels}
        if prediction not in matrix[target]:
            for values in matrix.values():
                values[prediction] = 0
            labels.append(prediction)
        matrix[target][prediction] += 1
    return matrix


def markdown_path(target: Path, *, base: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve())


def save_waterfall_plot(
    iq_path: Path,
    png_path: Path,
    *,
    row: dict[str, str],
    sample_rate: float,
    freq: float,
    nfft: int,
) -> bool:
    if not iq_path.exists():
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping waterfall PNG export.", file=sys.stderr)
        return False

    iq = load_iq_file(iq_path)
    complex_iq = iq_to_complex64(iq)
    if len(complex_iq) < nfft:
        complex_iq = np.pad(complex_iq, (0, nfft - len(complex_iq)), mode="constant")

    hop = max(1, nfft // 2)
    starts = np.arange(0, len(complex_iq) - nfft + 1, hop)
    max_columns = 900
    if len(starts) > max_columns:
        starts = starts[np.linspace(0, len(starts) - 1, max_columns).astype(int)]
    window = np.hanning(nfft).astype(np.float32)
    frames = np.stack([complex_iq[start : start + nfft] * window for start in starts], axis=0)
    spectrum = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1).T
    power = 20.0 * np.log10(np.abs(spectrum) + 1e-8)
    floor = np.percentile(power, 5)
    ceiling = np.percentile(power, 99.5)

    freq_axis_mhz = (np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate)) + freq) / 1e6
    time_axis_ms = starts / sample_rate * 1000.0
    extent = [float(time_axis_ms[0]), float(time_axis_ms[-1]), float(freq_axis_mhz[0]), float(freq_axis_mhz[-1])]

    prediction = row.get("prediction", "?")
    confidence = row.get("confidence", "?")
    target = row.get("target_class", "?")
    target_confidence = row.get("target_non_noise_confidence", "?")
    title = f"TX {target} -> RX {prediction} ({confidence})"
    subtitle = f"target confidence {target_confidence} | capture {row.get('capture_power_db', '?')} dB"

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    image = ax.imshow(
        power,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=floor,
        vmax=ceiling,
    )
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(
        0.012,
        0.955,
        subtitle,
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        va="top",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 5},
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RF frequency (MHz)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.012)
    colorbar.set_label("Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return True


def write_class_sweep_markdown(
    path: Path,
    *,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    classes: list[str],
    command: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for row in rows if row.get("passed") == "1")
    total = len(rows)
    accuracy = passed / total if total else 0.0
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    command_text = shlex.join(command)

    matrix = confusion_matrix_counts(rows, classes)
    matrix_labels = list(matrix.keys())
    for row_values in matrix.values():
        for label in row_values:
            if label not in matrix_labels:
                matrix_labels.append(label)

    lines = [
        "# Live OTA Noisy Drone RF Classification Results",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This test replays labeled NoisyDroneRF IQ samples over the air from one SDR and classifies the received "
        "live RF capture from another SDR. The result below is an end-to-end TX/RX hardware classification check, "
        "not just offline inference on dataset files.",
        "",
        "## Summary",
        "",
        f"- Trials: `{total}`",
        f"- Exact final prediction matches: `{passed}/{total}`",
        f"- Accuracy: `{accuracy:.3f}`",
        f"- Classes: `{', '.join(classes)}`",
        f"- CSV: `{args.tx_test_output_csv}`",
        f"- RX IQ windows: `{args.tx_test_save_rx_dir}`",
        f"- Waterfall snapshots: `{args.tx_test_save_plots_dir}`",
        "",
        "## OTA SDR Setup",
        "",
        "| Setting | Value |",
        "|---|---:|",
        f"| Model | `{markdown_escape(args.model)}` |",
        f"| TX SDR | `{markdown_escape(args.tx_device_args)}` |",
        f"| TX frontend | `{frontend_label(args.tx_device_args, 'TX', args.tx_channel)}` |",
        f"| TX antenna | `{markdown_escape(args.tx_antenna)}` |",
        f"| RX SDR | `{markdown_escape(args.rx_device_args)}` |",
        f"| RX frontend | `{frontend_label(args.rx_device_args, 'RX', args.rx_channel)}` |",
        f"| RX antenna | `{markdown_escape(args.rx_antenna)}` |",
        f"| Frequency | `{args.freq:.0f} Hz` |",
        f"| Sample rate | `{args.sample_rate:.0f} S/s` |",
        f"| Bandwidth | `{args.bandwidth:.0f} Hz` |",
        f"| RX gain | `{args.rx_gain}` |",
        f"| TX gain | `{args.tx_gain}` |",
        f"| TX amplitude | `{args.tx_amplitude}` |",
        f"| TX min SNR | `{args.tx_min_snr}` |",
        f"| Window samples | `{args.window_samples}` |",
        f"| Capture samples | `{args.capture_samples if args.capture_samples is not None else args.window_samples * 4}` |",
        f"| Window score mode | `{args.window_score_mode}` |",
        f"| Decision mode | `{args.decision_mode}` |",
        f"| Non-noise threshold | `{args.non_noise_threshold}` |",
        "",
        "## Confusion Matrix",
        "",
        "Rows are transmitted dataset labels. Columns are final live OTA predictions.",
        "",
        "| TX \\ RX | " + " | ".join(markdown_escape(label) for label in matrix_labels) + " |",
        "|---|" + "|".join("---:" for _ in matrix_labels) + "|",
    ]

    for target in matrix_labels:
        row_values = matrix.get(target, {})
        lines.append(
            "| "
            + markdown_escape(target)
            + " | "
            + " | ".join(str(row_values.get(prediction, 0)) for prediction in matrix_labels)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Waterfall Snapshots",
            "",
            "Each image is rendered from the selected live RX IQ window used for classification. The overlay shows "
            "the transmitted class, final prediction, confidence, and capture power.",
            "",
        ]
    )
    for row in rows:
        plot_path = row.get("waterfall_png")
        if not plot_path:
            continue
        lines.extend(
            [
                f"### Trial {markdown_escape(row.get('trial'))}: "
                f"{markdown_escape(row.get('target_class'))} -> {markdown_escape(row.get('prediction'))}",
                "",
                f"![Waterfall trial {markdown_escape(row.get('trial'))}]"
                f"({markdown_path(Path(plot_path), base=path.parent)})",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            command_text,
            "```",
            "",
            "## Per-Class Summary",
            "",
            "| Class | Pass/Total | Accuracy | Min Target Confidence | Mean Target Confidence | Mean Capture Power dB | Max Full-Scale % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for item in summarize_class_rows(rows):
        lines.append(
            "| "
            f"{markdown_escape(item['class'])} | "
            f"{item['passed']}/{item['total']} | "
            f"{item['accuracy']} | "
            f"{item['min_target_confidence']} | "
            f"{item['mean_target_confidence']} | "
            f"{item['mean_capture_power_db']} | "
            f"{item['max_fullscale_pct']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Trial Results",
            "",
            "| Trial | Target | Prediction | Confidence | Best Non-Noise | Target Confidence | Capture Power dB | Full-Scale % | TX Sample | RX IQ | Waterfall |",
            "|---:|---|---|---:|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{markdown_escape(row.get('trial'))} | "
            f"{markdown_escape(row.get('target_class'))} | "
            f"{markdown_escape(row.get('prediction'))} | "
            f"{markdown_escape(row.get('confidence'))} | "
            f"{markdown_escape(row.get('best_non_noise'))} | "
            f"{markdown_escape(row.get('target_non_noise_confidence'))} | "
            f"{markdown_escape(row.get('capture_power_db'))} | "
            f"{markdown_escape(row.get('mag_fullscale_pct'))} | "
            f"{markdown_escape(row.get('tx_sample'))} | "
            f"`{markdown_escape(row.get('rx_iq_path'))}` | "
            f"`{markdown_escape(row.get('waterfall_png'))}` |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `prediction` is the final script decision after the configured decision policy.",
            "- `best_non_noise` and `target confidence` are conditional on the non-noise class mass.",
            "- Full-scale percentages above zero indicate some clipping or saturation in the saved RX window.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tx_class_sweep(args: argparse.Namespace) -> int:
    classes = [item.strip() for item in args.tx_test_classes.split(",") if item.strip()]
    invalid = [item for item in classes if item not in LABEL_NAMES]
    if invalid:
        print(f"Unknown class in --tx-test-classes: {', '.join(invalid)}", file=sys.stderr)
        return 2
    if args.tx_iq_file is not None:
        print("--tx-test-all-classes cannot be combined with --tx-iq-file.", file=sys.stderr)
        return 2
    if args.iq_file is not None:
        print("--tx-test-all-classes cannot be combined with --iq-file.", file=sys.stderr)
        return 2

    args.tx_test_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.tx_test_save_rx_dir.mkdir(parents=True, exist_ok=True)
    args.tx_test_save_plots_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial",
        "target_class",
        "returncode",
        "passed",
        "prediction",
        "confidence",
        "best_non_noise",
        "best_non_noise_confidence",
        "target_non_noise_confidence",
        "decision",
        "tx_sample",
        "tx_file_prediction",
        "tx_file_confidence",
        "tx_file_best_non_noise",
        "tx_file_non_noise_confidence",
        "selected_window_start",
        "raw_capture_samples",
        "capture_power_db",
        "capture_peak",
        "i_clip_pct",
        "q_clip_pct",
        "mag_fullscale_pct",
        "top",
        "rx_iq_path",
        "waterfall_png",
    ]

    rows: list[dict[str, str]] = []
    trial_number = 0
    total = len(classes) * max(1, args.tx_test_count)
    entrypoint = sys.argv[0] if sys.argv[0] else str(Path(__file__).resolve())
    sweep_command = [sys.executable, entrypoint, *sys.argv[1:]]
    for class_index, class_name in enumerate(classes):
        for repeat_index in range(max(1, args.tx_test_count)):
            trial_number += 1
            seed = args.tx_test_seed_start + class_index * 1000 + repeat_index
            rx_iq_path = args.tx_test_save_rx_dir / f"{trial_number:03d}_{class_name}.npy"
            print(
                f"\n=== class sweep trial {trial_number}/{total}: target={class_name} seed={seed} ===",
                flush=True,
            )

            cmd = [
                sys.executable,
                entrypoint,
                "--tx",
                "--once",
                "--tx-class-name",
                class_name,
                "--tx-seed",
                str(seed),
                "--save-rx-iq",
                str(rx_iq_path),
            ]
            add_cli_arg(cmd, "--model", args.model)
            add_cli_arg(cmd, "--rx-device-args", args.rx_device_args)
            add_cli_arg(cmd, "--rx-channel", args.rx_channel)
            add_cli_arg(cmd, "--rx-antenna", args.rx_antenna)
            add_cli_arg(cmd, "--freq", args.freq)
            add_cli_arg(cmd, "--sample-rate", args.sample_rate)
            add_cli_arg(cmd, "--bandwidth", args.bandwidth)
            add_cli_arg(cmd, "--rx-gain", args.rx_gain)
            add_cli_arg(cmd, "--rx-stream-args", args.rx_stream_args)
            add_cli_arg(cmd, "--window-samples", args.window_samples)
            add_cli_arg(cmd, "--capture-samples", args.capture_samples)
            add_cli_arg(cmd, "--burst-smooth-samples", args.burst_smooth_samples)
            add_cli_arg(cmd, "--scan-stride-samples", args.scan_stride_samples)
            add_cli_arg(cmd, "--window-score-mode", args.window_score_mode)
            add_cli_arg(cmd, "--chunk-samples", args.chunk_samples)
            add_cli_arg(cmd, "--pretrigger-chunks", args.pretrigger_chunks)
            add_cli_arg(cmd, "--calibration-chunks", args.calibration_chunks)
            add_cli_arg(cmd, "--threshold-db", args.threshold_db)
            add_cli_arg(cmd, "--absolute-threshold-db", args.absolute_threshold_db)
            add_cli_arg(cmd, "--trigger-timeout-sec", args.trigger_timeout_sec)
            add_cli_arg(cmd, "--nfft", args.nfft)
            add_cli_arg(cmd, "--hop", args.hop)
            add_cli_arg(cmd, "--time-bins", args.time_bins)
            add_cli_arg(cmd, "--phase-tta", args.phase_tta)
            add_cli_arg(cmd, "--top-k", args.top_k)
            add_cli_arg(cmd, "--decision-mode", args.decision_mode)
            add_cli_arg(cmd, "--non-noise-threshold", args.non_noise_threshold)
            add_cli_arg(cmd, "--signal-margin-db", args.signal_margin_db)
            add_cli_arg(cmd, "--tx-device-args", args.tx_device_args)
            add_cli_arg(cmd, "--tx-channel", args.tx_channel)
            add_cli_arg(cmd, "--tx-antenna", args.tx_antenna)
            add_cli_arg(cmd, "--tx-gain", args.tx_gain)
            add_cli_arg(cmd, "--tx-bandwidth", args.tx_bandwidth)
            add_cli_arg(cmd, "--tx-stream-args", args.tx_stream_args)
            add_cli_arg(cmd, "--tx-dataset-dir", args.tx_dataset_dir)
            add_cli_arg(cmd, "--tx-min-snr", args.tx_min_snr)
            add_cli_arg(cmd, "--tx-amplitude", args.tx_amplitude)
            add_cli_arg(cmd, "--tx-pad-sec", args.tx_pad_sec)
            add_cli_arg(cmd, "--tx-repeat", args.tx_repeat)
            add_cli_arg(cmd, "--tx-delay-sec", args.tx_delay_sec)
            add_cli_arg(cmd, "--tx-capture-mode", args.tx_capture_mode)
            add_cli_arg(cmd, "--rx-flush-chunks", args.rx_flush_chunks)
            if args.rx_agc:
                cmd.append("--rx-agc")
            if not args.scan_windows:
                cmd.append("--no-scan-windows")

            child = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert child.stdout is not None
            lines = []
            for line in child.stdout:
                print(line, end="", flush=True)
                lines.append(line.rstrip("\n"))
            returncode = child.wait()

            row = parse_child_trial_log(lines)
            row["trial"] = str(trial_number)
            row["target_class"] = class_name
            row["returncode"] = str(returncode)
            row["rx_iq_path"] = str(rx_iq_path)
            row["passed"] = "1" if returncode == 0 and row.get("prediction") == class_name else "0"
            png_path = args.tx_test_save_plots_dir / f"{trial_number:03d}_{class_name}_waterfall.png"
            if returncode == 0 and save_waterfall_plot(
                rx_iq_path,
                png_path,
                row=row,
                sample_rate=args.sample_rate,
                freq=args.freq,
                nfft=args.nfft,
            ):
                row["waterfall_png"] = str(png_path)
            rows.append(row)

            with args.tx_test_output_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            write_class_sweep_markdown(
                args.tx_test_output_md,
                rows=rows,
                args=args,
                classes=classes,
                command=sweep_command,
            )

            if returncode != 0 and args.stop_on_test_error:
                print(f"Stopping class sweep after failed trial {trial_number}.", file=sys.stderr)
                break
        else:
            continue
        break

    passed = sum(1 for row in rows if row.get("passed") == "1")
    print(f"\nClass sweep complete: {passed}/{len(rows)} exact final predictions matched target.", flush=True)
    print(f"CSV summary: {args.tx_test_output_csv}", flush=True)
    print(f"Markdown report: {args.tx_test_output_md}", flush=True)
    for row in rows:
        print(
            f"{row.get('target_class', '?'):>10s} -> prediction={row.get('prediction', '?'):<10s} "
            f"best_non_noise={row.get('best_non_noise', '?'):<10s} "
            f"target_conf={row.get('target_non_noise_confidence', '?')}",
            flush=True,
        )
    return 0 if all(row.get("returncode") == "0" for row in rows) else 1


def main() -> int:
    args = parse_args()
    if args.ignore_noise_class:
        args.decision_mode = "non-noise"
    if args.tx_test_all_classes:
        return run_tx_class_sweep(args)
    if args.tx and args.iq_file is not None:
        print("--tx cannot be combined with --iq-file RX replay.", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"Missing model: {args.model}", file=sys.stderr)
        return 2

    from tensorflow.keras.models import load_model

    model = load_model(args.model, compile=False)
    expected_shape = tuple(model.input_shape[1:])
    script_shape = (args.nfft, args.time_bins, 2)
    if expected_shape != script_shape:
        print(f"Model expects {expected_shape}; script preprocessing is configured for {script_shape}.", file=sys.stderr)
        return 2

    tx_sink: SoapyIqSink | None = None
    tx_worker: TxWorker | None = None
    source: FileIqSource | SoapyIqSource | None = None
    if args.tx:
        tx_path, tx_iq = choose_tx_sample(
            args.tx_dataset_dir,
            args.tx_iq_file,
            seed=args.tx_seed,
            min_snr=args.tx_min_snr,
            target=args.tx_target,
            class_name=args.tx_class_name,
        )
        tx_label, tx_confidence, tx_probs = classify_iq(
            model,
            tx_iq,
            nfft=args.nfft,
            hop=args.hop,
            time_bins=args.time_bins,
            labels=LABEL_NAMES,
            phase_tta=1,
        )
        tx_non_noise_label, tx_non_noise_confidence = best_non_noise_prediction(tx_probs, LABEL_NAMES)
        tx_iq = prepare_tx_iq(
            tx_iq,
            amplitude=args.tx_amplitude,
            pad_seconds=args.tx_pad_sec,
            sample_rate=args.sample_rate,
        )
        print(f"TX sample: {describe_noisy_drone_sample(tx_path)}", flush=True)
        print(
            f"TX file model prediction={tx_label} confidence={tx_confidence:.3f} "
            f"best_non_noise={tx_non_noise_label} non_noise_confidence={tx_non_noise_confidence:.3f}",
            flush=True,
        )
        print(
            f"TX prepared samples={len(tx_iq)} amplitude={args.tx_amplitude:.3f} "
            f"repeat={args.tx_repeat}",
            flush=True,
        )
        tx_sink = SoapyIqSink(
            device_args=args.tx_device_args,
            channel=args.tx_channel,
            sample_rate=args.sample_rate,
            freq=args.freq,
            gain=args.tx_gain,
            antenna=args.tx_antenna,
            bandwidth=args.tx_bandwidth if args.tx_bandwidth is not None else args.bandwidth,
            stream_args=args.tx_stream_args,
        )
        tx_worker = TxWorker(
            tx_sink,
            tx_iq,
            chunk_samples=args.chunk_samples,
            repeats=args.tx_repeat,
        )
        print(
            f"TX ready with SoapySDR device={args.tx_device_args!r} "
            f"frontend={frontend_label(args.tx_device_args, 'TX', args.tx_channel)} "
            f"antenna={args.tx_antenna!r} freq={args.freq:.0f} Hz",
            flush=True,
        )

    tx_started_before_rx = False
    if args.tx and args.tx_capture_mode == "direct" and tx_worker is not None:
        print("Starting TX replay before opening RX", flush=True)
        tx_worker.start()
        tx_started_before_rx = True
        if args.tx_delay_sec > 0:
            print(f"TX lead-in {args.tx_delay_sec:.3f}s before RX open", flush=True)
            time.sleep(args.tx_delay_sec)

    if args.iq_file:
        source = FileIqSource(load_iq_file(args.iq_file), loop=args.loop_file)
        print(f"Replaying IQ from {args.iq_file}", flush=True)
    else:
        source = SoapyIqSource(
            device_args=args.rx_device_args,
            channel=args.rx_channel,
            sample_rate=args.sample_rate,
            freq=args.freq,
            gain=args.rx_gain,
            agc=args.rx_agc,
            antenna=args.rx_antenna,
            bandwidth=args.bandwidth,
            stream_args=args.rx_stream_args,
        )
        print(
            f"Listening with SoapySDR device={args.rx_device_args!r} "
            f"frontend={frontend_label(args.rx_device_args, 'RX', args.rx_channel)} "
            f"antenna={args.rx_antenna!r} freq={args.freq:.0f} Hz sample_rate={args.sample_rate:.0f} S/s "
            f"agc={source.gain_mode} gain={source.actual_gain}",
            flush=True,
        )

    try:
        while True:
            floor_db = None
            def start_tx_after_calibration(_floor_db, _trigger_db) -> None:
                if tx_worker is None:
                    return
                if tx_started_before_rx:
                    return
                if args.tx_delay_sec > 0 and args.tx_capture_mode != "direct":
                    time.sleep(args.tx_delay_sec)
                print("Starting TX replay", flush=True)
                tx_worker.start()

            if args.tx and args.tx_capture_mode == "direct":
                capture_samples = args.capture_samples
                if capture_samples is None:
                    capture_samples = args.window_samples * 4 if args.tx else args.window_samples
                floor_db = calibrate_noise_floor(
                    source,
                    chunk_samples=args.chunk_samples,
                    chunks=args.calibration_chunks,
                )
                print(f"Noise floor {floor_db:.1f} dB", flush=True)
                if args.tx_delay_sec > 0:
                    print(f"Settling {args.tx_delay_sec:.3f}s before final RX flush", flush=True)
                    time.sleep(args.tx_delay_sec)
                if args.rx_flush_chunks:
                    print(f"Flushing {args.rx_flush_chunks} RX chunks before TX capture", flush=True)
                    flush_rx(source, chunk_samples=args.chunk_samples, chunks=args.rx_flush_chunks)
                if not tx_started_before_rx:
                    start_tx_after_calibration(floor_db, floor_db + args.threshold_db)
                capture = capture_fixed_window(
                    source,
                    chunk_samples=args.chunk_samples,
                    capture_samples=capture_samples,
                )
            else:
                capture, floor_db, _ = capture_on_energy(
                    source,
                    chunk_samples=args.chunk_samples,
                    window_samples=args.window_samples,
                    threshold_db=args.threshold_db,
                    calibration_chunks=args.calibration_chunks,
                    absolute_threshold_db=args.absolute_threshold_db,
                    pretrigger_chunks=args.pretrigger_chunks,
                    on_calibrated=start_tx_after_calibration if args.tx else None,
                    trigger_timeout_sec=args.trigger_timeout_sec if args.trigger_timeout_sec is not None else (15.0 if args.tx else None),
                )
            if capture is None:
                if tx_worker is not None:
                    tx_worker.stop()
                    tx_worker.join(timeout=0.0)
                print("No signal captured before timeout or no more IQ samples available.", flush=True)
                return 0
            if tx_worker is not None:
                tx_worker.stop()
                tx_worker.join(timeout=1.0)
            raw_capture = capture
            probs = None
            if args.scan_windows and len(raw_capture) > args.window_samples:
                score_mode = args.window_score_mode
                if score_mode == "auto":
                    score_mode = "target" if args.tx and args.tx_class_name else "non-noise"
                target_idx = LABEL_NAMES.index(args.tx_class_name) if args.tx_class_name in LABEL_NAMES else None
                best = None
                for start in candidate_window_starts(
                    len(raw_capture),
                    args.window_samples,
                    args.scan_stride_samples,
                ):
                    candidate = raw_capture[start : start + args.window_samples]
                    cand_label, cand_confidence, cand_probs = classify_iq(
                        model,
                        candidate,
                        nfft=args.nfft,
                        hop=args.hop,
                        time_bins=args.time_bins,
                        labels=LABEL_NAMES,
                        phase_tta=args.phase_tta,
                    )
                    cand_non_noise_label, cand_non_noise_confidence = best_non_noise_prediction(cand_probs, LABEL_NAMES)
                    if score_mode == "target" and target_idx is not None:
                        score = conditional_class_confidence(cand_probs, LABEL_NAMES, target_idx)
                        score_detail = f"target={LABEL_NAMES[target_idx]}:{score:.3f}"
                    elif score_mode == "non-noise":
                        score = float(cand_non_noise_confidence)
                        score_detail = f"non_noise:{score:.3f}"
                    else:
                        score = float(cand_confidence)
                        score_detail = f"raw:{score:.3f}"
                    print(
                        f"candidate start={start} score={score:.3f} {score_detail} "
                        f"raw={cand_label}:{cand_confidence:.3f} "
                        f"best_non_noise={cand_non_noise_label}:{cand_non_noise_confidence:.3f}",
                        flush=True,
                    )
                    if best is None or score > best[0]:
                        best = (score, start, candidate, cand_label, cand_confidence, cand_probs)
                assert best is not None
                _, burst_start, capture, label, confidence, probs = best
                print(
                    f"Selected scanned model window start={burst_start} "
                    f"from raw_capture_samples={len(raw_capture)} mode={score_mode}",
                    flush=True,
                )
            else:
                capture, burst_start = select_high_power_window(
                    raw_capture,
                    window_samples=args.window_samples,
                    smooth_samples=args.burst_smooth_samples,
                )
                if len(raw_capture) != len(capture):
                    print(
                        f"Selected high-power model window start={burst_start} "
                        f"from raw_capture_samples={len(raw_capture)}",
                        flush=True,
                    )
            stats = capture_stats(capture)
            print(
                f"Capture power {stats['power_db']:.1f} dB, peak {stats['peak']:.3f}, "
                f"I clip {stats['i_clip_pct']:.2f}%, Q clip {stats['q_clip_pct']:.2f}%, "
                f"|IQ|>=1 {stats['mag_fullscale_pct']:.2f}%",
                flush=True,
            )
            if stats["i_clip_pct"] > 0.1 or stats["q_clip_pct"] > 0.1 or stats["mag_fullscale_pct"] > 0.5:
                print(
                    "Warning: RX capture is clipping. Lower --tx-amplitude, --tx-gain, or --rx-gain, "
                    "or add RF attenuation/coupling loss.",
                    flush=True,
                )
            if args.save_rx_iq is not None:
                args.save_rx_iq.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.save_rx_iq, capture.astype(np.float32))
                print(f"Saved RX IQ window to {args.save_rx_iq}", flush=True)
            if probs is None:
                label, confidence, probs = classify_iq(
                    model,
                    capture,
                    nfft=args.nfft,
                    hop=args.hop,
                    time_bins=args.time_bins,
                    labels=LABEL_NAMES,
                    phase_tta=args.phase_tta,
                )
            non_noise_label, non_noise_confidence = best_non_noise_prediction(probs, LABEL_NAMES)
            print(
                f"best_non_noise={non_noise_label} non_noise_confidence={non_noise_confidence:.3f}",
                flush=True,
            )
            if args.tx and args.tx_class_name in LABEL_NAMES and args.tx_class_name != "Noise":
                target_idx = LABEL_NAMES.index(args.tx_class_name)
                target_confidence = conditional_class_confidence(probs, LABEL_NAMES, target_idx)
                print(
                    f"target_class={args.tx_class_name} target_non_noise_confidence={target_confidence:.3f}",
                    flush=True,
                )
            signal_present = bool(args.tx)
            if floor_db is not None:
                signal_present = signal_present or stats["power_db"] >= floor_db + args.signal_margin_db
            label, confidence, decision_detail = choose_final_prediction(
                label,
                confidence,
                probs,
                LABEL_NAMES,
                decision_mode=args.decision_mode,
                non_noise_threshold=args.non_noise_threshold,
                signal_present=signal_present,
                target_label=args.tx_class_name if args.tx else None,
            )
            if decision_detail is not None:
                print(decision_detail, flush=True)
            print_prediction(label, confidence, probs, LABEL_NAMES, top_k=args.top_k)
            if args.once:
                return 0
            time.sleep(args.cooldown_sec)
    finally:
        if tx_worker is not None:
            tx_worker.stop()
            tx_worker.join(timeout=1.0)
        if source is not None:
            source.close()
        if tx_sink is not None:
            tx_sink.close()


if __name__ == "__main__":
    raise SystemExit(main())
