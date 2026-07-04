# Live OTA RML2016 Modulation Classification Results

Generated: `2026-07-04T15:16:58+00:00`

This test replays labeled RML2016.10a IQ samples over the air from one SDR and classifies the received live RF capture from another SDR. The result below is an end-to-end TX/RX hardware classification check, not just offline inference on dataset files.

## Summary

- Trials: `1`
- Exact final prediction matches: `1/1`
- Accuracy: `1.000`
- Classes: `8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM`
- CSV: `../../outputs/rml2016_class_sweep.csv`
- RX IQ windows: `../../outputs/rml2016_class_sweep_iq`
- Waterfall snapshots: `../../outputs/rml2016_class_sweep_plots`

## OTA SDR Setup

| Setting | Value |
|---|---:|
| Model | `models/rml2016/rml2016_lstm_rnn_2024.keras` |
| TX SDR | `driver=bladerf,serial=7faa712b1fab42f4b84e494171b91721` |
| TX frontend | `bladeRF TX1` |
| TX antenna | `TX` |
| RX SDR | `driver=hackrf` |
| RX frontend | `RX channel 0` |
| RX antenna | `` |
| Frequency | `2399000000 Hz` |
| Sample rate | `20000000 S/s` |
| Bandwidth | `20000000 Hz` |
| RX gain | `60.0` |
| TX gain | `60.0` |
| TX amplitude | `0.2` |
| TX tile samples | `65536` |
| TX min SNR | `18` |
| Window samples | `128` |
| Capture samples | `8192` |
| Scan stride samples | `16` |
| Window score mode | `auto` |
| Decision mode | `hybrid` |
| Non-noise threshold | `0.55` |
| RML SNR feature | `18.0` |

## Confusion Matrix

Rows are transmitted dataset labels. Columns are final live OTA predictions.

| TX \ RX | 8PSK | AM-DSB | AM-SSB | BPSK | CPFSK | GFSK | PAM4 | QAM16 | QAM64 | QPSK | WBFM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AM-DSB | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AM-SSB | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BPSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CPFSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GFSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PAM4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QAM16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QAM64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QPSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| WBFM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Waterfall Snapshots

Each image is rendered from the selected live RX IQ window used for classification. The overlay shows the transmitted class, final prediction, confidence, and capture power.

### Trial 1: 8PSK -> 8PSK

![Waterfall trial 1](../../outputs/rml2016_class_sweep_plots/001_8PSK_waterfall.png)


## Command

```bash
/home/jake/workspace/SDR/RF_Sentinel/.venv/bin/python3 /home/jake/workspace/SDR/rf-signal-intelligence/scripts/live_rml2016_rf_classifier.py --tx-test-all-classes --tx-test-count 3
```

## Per-Class Summary

| Class | Pass/Total | Accuracy | Min Target Confidence | Mean Target Confidence | Mean Capture Power dB | Max Full-Scale % |
|---|---:|---:|---:|---:|---:|---:|
| 8PSK | 1/1 | 1.000 | 0.477 | 0.477 | -14.5 | 0.000 |

## Per-Trial Results

| Trial | Target | Prediction | Confidence | Best Non-Noise | Target Confidence | Capture Power dB | Full-Scale % | TX Sample | RX IQ | Waterfall |
|---:|---|---|---:|---|---:|---:|---:|---|---|---|
| 1 | 8PSK | 8PSK | 0.477 | 8PSK | 0.477 | -14.5 | 0.00 | RML2016_8PSK_snr18_sample685 true=8PSK target=0 snr=18dB | `../../outputs/rml2016_class_sweep_iq/001_8PSK.npy` | `../../outputs/rml2016_class_sweep_plots/001_8PSK_waterfall.png` |

## Notes

- `prediction` is the final script decision after the configured decision policy.
- `best_non_noise` and `target confidence` are conditional on the non-noise class mass.
- Full-scale percentages above zero indicate some clipping or saturation in the saved RX window.
