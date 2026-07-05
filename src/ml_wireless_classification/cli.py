"""Compatibility entrypoint for the renamed `rf_signal_intelligence.cli` module."""

from __future__ import annotations

from rf_signal_intelligence.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
