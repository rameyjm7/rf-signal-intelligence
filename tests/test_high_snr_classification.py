import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from tests.integration_policy import require_paths

MODEL_CANDIDATES = [
    "rml2016/rml2016_lstm_rnn_2024.keras",
    "rml2016/rml2016_rnn_lstm_with_snr_5_2_1.keras",
]
SAMPLES_PER_CLASS = 25
MIN_HIGH_SNR_ACCURACY = 0.70


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rml2016_data(data_path: Path):
    with data_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _build_max_snr_batch(data: dict):
    max_snr = max(snr for (_, snr) in data.keys())
    x_rows = []
    labels = []

    for (modulation, snr), signals in data.items():
        if snr != max_snr:
            continue

        limit = min(SAMPLES_PER_CLASS, len(signals))
        for idx in range(limit):
            signal = signals[idx]
            iq_signal = np.vstack([signal[0], signal[1]]).T
            snr_feature = np.full((iq_signal.shape[0], 1), snr)
            combined = np.hstack([iq_signal, snr_feature])
            x_rows.append(combined)
            labels.append(modulation)

    x = np.asarray(x_rows, dtype=np.float32)
    return x, labels, max_snr


def _resolve_compatible_model(models_dir: Path):
    errors = {}
    for model_name in MODEL_CANDIDATES:
        model_path = models_dir / model_name
        if not model_path.exists():
            errors[model_name] = "missing file"
            continue

        try:
            model = load_model(model_path, compile=False)
        except Exception as exc:
            errors[model_name] = str(exc).splitlines()[0]
            continue

        input_shape = tuple(model.input_shape[1:])
        if input_shape == (128, 3):
            return model_name, model

        errors[model_name] = f"unexpected input shape {input_shape}"

    return None, errors


@pytest.mark.integration
def test_max_snr_samples_are_classified_with_strong_accuracy():
    repo_root = _repo_root()
    data_path = repo_root / "data" / "RML2016" / "RML2016.10a_dict.pkl"
    models_dir = repo_root / "models"
    require_paths("RML2016", [data_path, models_dir])

    data = _load_rml2016_data(data_path)
    x_max_snr, labels, max_snr = _build_max_snr_batch(data)
    assert len(labels) > 0, "No highest-SNR samples were collected from the dataset."

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted({modulation for (modulation, _) in data.keys()}))
    y_true = label_encoder.transform(labels)

    model_name, model_or_errors = _resolve_compatible_model(models_dir)
    if model_name is None:
        pytest.skip(
            "No compatible RML2016 model could be loaded. "
            f"Model resolution details: {model_or_errors}"
        )
    model = model_or_errors
    predictions = model.predict(x_max_snr, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy >= MIN_HIGH_SNR_ACCURACY, (
        f"High-SNR accuracy too low for {model_name} at SNR={max_snr} dB: {accuracy:.3f} "
        f"(threshold: {MIN_HIGH_SNR_ACCURACY:.2f}, n={len(y_true)})"
    )
