# Result Card: Background False-Positive Rejection

## Objective

Show that the deployed NoisyDroneRF classifier rejects non-drone/background RF and does not raise drone alerts during representative no-drone conditions.

## Setup and Hardware

- Receiver: Jetson + bladeRF through local `sdr-gateway`
- Model backend: TensorRT FP16 NoisyDroneRF classifier
- Event path: gateway IQ -> private framing/preprocessing -> TensorRT inference -> quality/confidence gates -> structured events
- Tested conditions: Wi-Fi-heavy 2.4 GHz, Bluetooth-heavy 2.4 GHz, random 2.4 GHz background, and longer no-drone soak

## Method

Each condition captured live RF through the Jetson receiver path and generated structured RFML events. A detection counted as an alert only when the final event type was a drone detection; background rejection expected a no-alert event.

## Metrics

| Condition | Events | Alerts | No-alerts | Alert rate | Mean SNR | Mean E2E | Mean inference | Top predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Wi-Fi-heavy 2412 MHz no-drone | 30 | 0 | 30 | 0.0% | 10.6 dB | 366.1 ms | 122.8 ms | Noise:30 |
| Bluetooth-heavy 2402 MHz no-drone | 30 | 0 | 30 | 0.0% | 10.5 dB | 358.9 ms | 124.5 ms | Noise:30 |
| Random 2440 MHz no-drone | 30 | 0 | 30 | 0.0% | 11.3 dB | 358.9 ms | 122.7 ms | Noise:30 |
| 2470 MHz no-drone soak | 150 | 0 | 150 | 0.0% | 12.2 dB | 350.8 ms | 122.4 ms | Noise:150 |

Aggregate: 240 background events, 0 drone alerts.

## Limitations

- This is a local RF environment test, not an exhaustive environmental survey.
- No physical DJI Mini flight was present; this card only covers no-drone/background rejection.
- Background conditions are representative of the available lab/office spectrum during the run.

## Artifacts

- `results/noisy_drone_rf_v2/background_2470_false_positive_summary.md`
- `results/noisy_drone_rf_v2/background_matrix/expanded_background_matrix_summary.md`
- Private archive: raw event logs and replay/capture artifacts.
