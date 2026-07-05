"""Compatibility package for the renamed `rf_signal_intelligence` package."""

from __future__ import annotations

import importlib
import sys

_package = importlib.import_module("rf_signal_intelligence")
sys.modules[__name__] = _package

