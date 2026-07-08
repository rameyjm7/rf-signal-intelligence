# Result Card: Multi-Class OTA Positive Control

## Objective

Demonstrate proposal-ready NoisyDroneRF classification coverage by replaying representative high-SNR dataset IQ clips over the air for every non-noise class and verifying the deployed Jetson receiver reports the expected class.

## Setup and Hardware

- RX: Jetson bladeRF via local `sdr-gateway`
- TX: local bladeRF replaying NoisyDroneRF dataset IQ files
- Frequency: 2470 MHz
- Sample rate / bandwidth: 20 MHz / 20 MHz
- Backend: TensorRT FP16
- Reports per phase: 3
- Gates: `min_snr_db=5`, `min_detection_confidence=0.9`

## Method

The session ran no-TX baseline, six class-specific OTA replay phases, and no-TX recovery. Each report emitted JSONL and saved received IQ for regression replay.

## Metrics

| Phase | Expected | Reports | Correct | Alerts | No-alerts | Predictions | Result |
|---|---|---:|---:|---:|---:|---|---|
| Background before TX | Noise | 3 | 3 | 0 | 3 | Noise:3 | PASS |
| DJI positive control | DJI | 3 | 3 | 3 | 0 | DJI:3 | PASS |
| FutabaT14 positive control | FutabaT14 | 3 | 3 | 3 | 0 | FutabaT14:3 | PASS |
| FutabaT7 positive control | FutabaT7 | 3 | 3 | 3 | 0 | FutabaT7:3 | PASS |
| Graupner positive control | Graupner | 3 | 3 | 3 | 0 | Graupner:3 | PASS |
| Taranis positive control | Taranis | 3 | 3 | 3 | 0 | Taranis:3 | PASS |
| Turnigy positive control | Turnigy | 3 | 3 | 3 | 0 | Turnigy:3 | PASS |
| Background after TX | Noise | 3 | 3 | 0 | 3 | Noise:3 | PASS |

Aggregate: 24/24 reports matched expected behavior; 18/18 positive controls were correctly classified; 6/6 no-TX controls produced no alert.

## Limitations

- Controlled OTA replay of labeled dataset IQ, not live flight testing with physical controllers.
- The result validates this deployed RF chain and gating configuration at 2470 MHz / 20 MHz.
- Raw IQ evidence is local and ignored by Git due to size.

## Artifacts

- `results/noisy_drone_rf_v2/multiclass_pc_20260708T145502Z/proposal_report.md`
- `results/noisy_drone_rf_v2/multiclass_pc_20260708T145502Z/proposal_table.csv`
- `results/noisy_drone_rf_v2/multiclass_pc_20260708T145502Z/event_level_results.csv`
- `results/noisy_drone_rf_v2/multiclass_pc_20260708T145502Z/replay_manifest.json`
- Local replay IQ: `results/noisy_drone_rf_v2/multiclass_pc_20260708T145502Z/*/iq/*.npy`
