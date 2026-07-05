"""Compatibility wrapper for archived ensemble notebook imports.

The legacy import target now lives in `ml_wireless_classification.legacy`.
"""

from ml_wireless_classification.legacy.rnn_lstm_w_snr_ensemble import ModulationLSTMClassifier

__all__ = ["ModulationLSTMClassifier"]
