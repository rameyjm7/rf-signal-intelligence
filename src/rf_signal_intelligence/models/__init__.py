"""Maintained model architectures."""

from rf_signal_intelligence.models.deepradar_cnn_bilstm import build_deepradar_cnn_bilstm
from rf_signal_intelligence.models.rml2016_lstm import build_rml2016_lstm_model
from rf_signal_intelligence.models.rml2018_lstm import (
    auto_select_batch_size,
    build_rml2018_lstm_model,
)

__all__ = [
    "ModulationLSTMClassifier",
    "auto_select_batch_size",
    "build_deepradar_cnn_bilstm",
    "build_rml2016_lstm_model",
    "build_rml2018_lstm_model",
]


def __getattr__(name: str):
    if name == "ModulationLSTMClassifier":
        from rf_signal_intelligence.models.rnn_lstm_with_snr import ModulationLSTMClassifier

        return ModulationLSTMClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
