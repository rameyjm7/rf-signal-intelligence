"""Legacy compatibility shim.

Archive notebooks historically imported `ModulationLSTMClassifier` from this module.
The maintained implementation now lives in
`ml_wireless_classification.models.rnn_lstm_with_snr`.
"""

from ml_wireless_classification.models.rnn_lstm_with_snr import ModulationLSTMClassifier

__all__ = ["ModulationLSTMClassifier"]
