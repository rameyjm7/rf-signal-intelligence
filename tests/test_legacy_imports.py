def test_random_forest_legacy_import_paths_match():
    from rf_signal_intelligence.legacy.random_forest_modulation_classifier import (
        RandomForestModulationClassifier as LegacyRandomForestModulationClassifier,
    )
    from rf_signal_intelligence.random_forest_modulation_classifier import (
        RandomForestModulationClassifier,
    )

    assert RandomForestModulationClassifier is LegacyRandomForestModulationClassifier


def test_rnn_lstm_ensemble_legacy_import_paths_match():
    from rf_signal_intelligence.legacy.rnn_lstm_w_snr_ensemble import (
        ModulationLSTMClassifier as LegacyModulationLSTMClassifier,
    )
    from rf_signal_intelligence.rnn_lstm_w_SNR_ensemble import ModulationLSTMClassifier

    assert ModulationLSTMClassifier is LegacyModulationLSTMClassifier
