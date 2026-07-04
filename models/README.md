# Model Zoo

This directory is organized by dataset to keep artifacts discoverable and reduce naming ambiguity.

## Layout

```text
models/
  rml2016/
    rml2016_rnn_lstm_with_snr_5_2_1.keras
    rml2016_lstm_rnn_2024.keras
    rml2016_gan_model_9270.keras
  rml2018/
    rml2018_lstm_rnn.keras
    rml2018_lstm_balanced.keras
    checkpoints/
      rml2018_lstm_finetuned_*.keras
      *.json
      best_checkpoint.txt
  deepradar2022/
    deepradar2022_cnn_bilstm_final.keras
  noisy_drone_rf_v2/
    noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras
    noisy_drone_rf_v2_vgg_full_complex_spectrogram_final.keras
```

## Models

| Path | Dataset | Architecture | Notes |
|---|---|---|---|
| `models/rml2016/rml2016_rnn_lstm_with_snr_5_2_1.keras` | RML2016.10a | RNN-LSTM (SNR-augmented) | Main RML2016 model with SNR appended to I/Q channels. |
| `models/rml2016/rml2016_lstm_rnn_2024.keras` | RML2016.10a | Stacked LSTM-RNN | Updated 2024 baseline model. |
| `models/rml2016/rml2016_gan_model_9270.keras` | RML2016.10a | GAN-assisted | Experimental GAN-derived classifier. |
| `models/rml2018/rml2018_lstm_rnn.keras` | RML2018.01A | LSTM-RNN | Primary RML2018 model. |
| `models/rml2018/rml2018_lstm_balanced.keras` | RML2018.01A | LSTM-RNN | Class-balancing variant. |
| `models/deepradar2022/deepradar2022_cnn_bilstm_final.keras` | DeepRadar2022 | CNN + BiLSTM | Final radar waveform model. |
| `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras` | Noisy Drone RF v2 | VGG full-complex spectrogram | Canonical Noisy Drone RF v2 model used by notebooks `33`, `44`, and `50`; large model artifact is expected locally and is not currently committed on this branch. |
| `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_final.keras` | Noisy Drone RF v2 | VGG full-complex spectrogram | Latest continuation checkpoint from notebook `33`; use the best checkpoint above for canonical reported metrics. |

## Usage

```python
from tensorflow.keras.models import load_model

model = load_model("models/rml2016/rml2016_lstm_rnn_2024.keras")
```

## Naming Convention

- Use snake_case.
- Prefix with dataset name (`rml2016_`, `rml2018_`, `deepradar2022_`).
- Keep architecture and variant information in filename suffix.
