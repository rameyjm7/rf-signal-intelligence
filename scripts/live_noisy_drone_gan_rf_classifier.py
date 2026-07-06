#!/usr/bin/env python3
"""Generate NoisyDroneRFv2 GAN I/Q, transmit it over SDR, and classify live RX."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    from rf_signal_intelligence.live_noisy_drone_rf_classifier import main as live_main

    return live_main()


if __name__ == "__main__":
    raise SystemExit(main())
