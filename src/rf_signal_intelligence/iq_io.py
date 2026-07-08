"""Generic IQ file loading helpers for bring-your-own-model workflows."""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np


class TorchStorageRef:
    """Minimal tensor storage reference for reading simple zip-based `.pt` tensors."""

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
    """Read simple tensor `.pt` archives without importing torch."""

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


def coerce_iq_array(value: object) -> np.ndarray:
    """Return IQ as float32 rows of `[I, Q]`."""

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


def load_iq_file(path: str | Path) -> np.ndarray:
    """Load common IQ file formats into `[I, Q]` float32 rows."""

    path = Path(path)
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
