"""Core training/runtime components for maintained workflows."""

from __future__ import annotations

__all__ = ["BaseModulationClassifier", "CustomEarlyStopping", "RUN_MODE", "CommonVars", "common_vars"]


def __getattr__(name: str):
    if name == "BaseModulationClassifier":
        from rf_signal_intelligence.core.base_classifier import BaseModulationClassifier

        return BaseModulationClassifier
    if name == "CustomEarlyStopping":
        from rf_signal_intelligence.core.callbacks import CustomEarlyStopping

        return CustomEarlyStopping
    if name in {"RUN_MODE", "CommonVars", "common_vars"}:
        from rf_signal_intelligence.core.run_mode import RUN_MODE, CommonVars, common_vars

        return {"RUN_MODE": RUN_MODE, "CommonVars": CommonVars, "common_vars": common_vars}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
