#!/usr/bin/env python3
"""CLI wrapper for the NoisyDroneRF live SDR classifier."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml_wireless_classification.live_noisy_drone_rf_classifier import main


if __name__ == "__main__":
    raise SystemExit(main())
