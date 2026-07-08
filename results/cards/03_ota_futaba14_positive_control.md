# Result Card: OTA FutabaT14 Positive Control With Replay IQ

## Objective

Verify that the Jetson RFML receiver detects a known labeled drone-controller signal over the air, while rejecting background before and after transmission. Preserve received IQ for later replay/regression.

## Setup and Hardware

- RX: Jetson bladeRF via local `sdr-gateway`
- TX: local bladeRF replaying labeled NoisyDroneRF dataset IQ
- Test class: FutabaT14
- Frequency: 2470 MHz
- Sample rate / bandwidth: 20 MHz / 20 MHz
- RX gates: `min_snr_db=5`, `min_detection_confidence=0.9`

## Method

The session used three phases: no-TX background, FutabaT14 OTA file replay, and no-TX recovery. Each classified report saved the received IQ capture to `report_*.npy` and emitted a JSON event.

## Metrics

| Phase | Events | Expected | Correct | Alerts | No-alerts | Top predictions |
|---|---:|---|---:|---:|---:|---|
| Background before TX | 4 | Noise/no-alert | 4 | 0 | 4 | Noise:4 |
| FutabaT14 positive control | 4 | FutabaT14/detection | 4 | 4 | 0 | FutabaT14:4 |
| Background after TX | 4 | Noise/no-alert | 4 | 0 | 4 | Noise:4 |

Aggregate: 12/12 reports matched expected behavior.

## Limitations

- This is OTA replay of a labeled dataset clip, not a live physical Futaba controller test.
- Confidence gating is part of the result and should be preserved in operational settings unless revalidated.
- Raw IQ captures are large and intentionally kept out of Git.

## Artifacts

- `results/noisy_drone_rf_v2/fn_pc_gated_20260708T143312Z/result_card.md`
- `results/noisy_drone_rf_v2/fn_pc_gated_20260708T143312Z/replay_manifest.json`
- `results/noisy_drone_rf_v2/fn_pc_gated_20260708T143312Z/*/events.jsonl`
- Local replay IQ: `results/noisy_drone_rf_v2/fn_pc_gated_20260708T143312Z/*/iq/*.npy`
