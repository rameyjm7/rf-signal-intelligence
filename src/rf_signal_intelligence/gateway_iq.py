"""sdr-gateway IQ source helpers for RFML validation and integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from websocket import create_connection


@dataclass(frozen=True)
class GatewayStreamConfig:
    base_url: str = "http://127.0.0.1:8080"
    device_id: str = "bladerf:0"
    center_freq_hz: int = 2_470_000_000
    sample_rate_sps: int = 20_000_000
    bandwidth_hz: int = 20_000_000
    lna_gain_db: int = 32
    vga_gain_db: int = 40
    amp_enable: bool = True
    rx_channel: int = 0
    replace_existing: bool = True
    iq_format: str = "i8"
    token: str | None = None
    timeout_sec: float = 5.0


class GatewayIqSource:
    """Read complex IQ from an sdr-gateway websocket stream."""

    def __init__(self, config: GatewayStreamConfig) -> None:
        self.config = config
        self.api_base = config.base_url.rstrip("/")
        self.ws_base = self.api_base.replace("http://", "ws://").replace("https://", "wss://", 1)
        self.stream_id: str | None = None
        self.iq_format = config.iq_format
        self.rx_channels = [int(config.rx_channel)]
        self._pending: deque[np.ndarray] = deque()
        self._pending_offset = 0
        self._ws = None
        self._start_stream()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.token}"} if self.config.token else {}

    def _start_stream(self) -> None:
        payload: dict[str, Any] = {
            "device_id": self.config.device_id,
            "center_freq_hz": int(self.config.center_freq_hz),
            "sample_rate_sps": int(self.config.sample_rate_sps),
            "lna_gain_db": int(self.config.lna_gain_db),
            "vga_gain_db": int(self.config.vga_gain_db),
            "amp_enable": bool(self.config.amp_enable),
            "replace_existing": bool(self.config.replace_existing),
            "baseband_filter_hz": int(self.config.bandwidth_hz),
            "rx_channels": list(self.rx_channels),
            "iq_format": self.config.iq_format,
        }
        response = requests.post(
            f"{self.api_base}/streams/start",
            json=payload,
            headers=self._headers(),
            timeout=self.config.timeout_sec,
        )
        response.raise_for_status()
        state = response.json()
        self.stream_id = str(state["stream_id"])
        stream_config = state.get("config") or {}
        self.iq_format = str(stream_config.get("iq_format") or self.config.iq_format or "i8").lower()
        channels = stream_config.get("rx_channels")
        if isinstance(channels, list) and channels:
            self.rx_channels = [int(ch) for ch in channels]

        ws_headers = [f"Authorization: Bearer {self.config.token}"] if self.config.token else None
        self._ws = create_connection(
            f"{self.ws_base}/ws/iq/{self.stream_id}?start=latest&keep=0",
            header=ws_headers,
            timeout=self.config.timeout_sec,
            enable_multithread=True,
        )

    def read_iq(self, count: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        remaining = int(count)
        while remaining > 0:
            pending = self._pop_pending(remaining)
            if pending.size:
                chunks.append(pending)
                remaining -= int(pending.size)
                continue
            assert self._ws is not None
            frame = self._ws.recv()
            if not isinstance(frame, (bytes, bytearray)):
                continue
            decoded = self._decode_frame(frame)
            if decoded.size:
                self._pending.append(decoded)
        return np.concatenate(chunks).astype(np.complex64, copy=False)

    def read_iq_pairs(self, count: int) -> np.ndarray:
        iq = self.read_iq(count)
        return np.column_stack((iq.real, iq.imag)).astype(np.float32, copy=False)

    def _pop_pending(self, count: int) -> np.ndarray:
        if not self._pending:
            return np.empty(0, dtype=np.complex64)
        first = self._pending[0]
        start = self._pending_offset
        end = min(start + count, first.size)
        chunk = first[start:end]
        if end >= first.size:
            self._pending.popleft()
            self._pending_offset = 0
        else:
            self._pending_offset = end
        return chunk

    def _decode_frame(self, frame: bytes | bytearray) -> np.ndarray:
        if self.iq_format == "cf32":
            values = np.frombuffer(frame, dtype=np.complex64)
            return values.astype(np.complex64, copy=False)
        if self.iq_format == "cs16":
            raw = np.frombuffer(frame, dtype=np.int16)
            scale = 32768.0
        else:
            raw = np.frombuffer(frame, dtype=np.int8)
            scale = 128.0
        values_per_complex = 2
        channel_count = max(1, len(self.rx_channels))
        values_per_sample = values_per_complex * channel_count
        usable = raw.size - (raw.size % values_per_sample)
        if usable <= 0:
            return np.empty(0, dtype=np.complex64)
        raw = raw[:usable]
        if channel_count > 1:
            raw = raw.reshape(-1, channel_count, values_per_complex)[:, 0, :].reshape(-1)
        if raw.size % 2:
            raw = raw[:-1]
        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        return ((i + 1j * q) / scale).astype(np.complex64, copy=False)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self.stream_id:
            try:
                requests.post(
                    f"{self.api_base}/streams/{self.stream_id}/stop",
                    headers=self._headers(),
                    timeout=self.config.timeout_sec,
                )
            except Exception:
                pass
            self.stream_id = None

    def __enter__(self) -> GatewayIqSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
