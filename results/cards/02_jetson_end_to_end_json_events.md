# Result Card: Jetson End-to-End RFML JSON Event Pipeline

## Objective

Show the deployed Jetson RFML path operating end-to-end from live/recorded SDR IQ through preprocessing, TensorRT inference, and structured JSON event output with measured latency.

## Setup and Hardware

- Receiver: Jetson + bladeRF exposed through local `sdr-gateway`
- Runtime: private gateway classifier using TensorRT FP16 engine
- Model: NoisyDroneRFv2 VGG/full-complex-spectrogram classifier
- Input rate: 20 MHz sample rate / 20 MHz bandwidth for RFML captures
- Output: structured events with prediction, gates, signal quality, selected window, and timing fields

## Method

The private gateway classifier captures IQ from `sdr-gateway`, discards startup frames, builds candidate model windows, runs TensorRT inference, applies quality/confidence gates, and emits one compact JSON event per report. Event timing captures read latency, preprocessing, inference, decision/gating, and total end-to-end latency.

## Metrics

| Evidence run | Events | Alerts | No-alerts | Mean SNR | Mean E2E | Mean inference | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| Background 2470 MHz no-drone | 30 | 0 | 30 | 14.1 dB | 363.2 ms | 124.0 ms | structured events |
| Multi-class OTA proposal run | 24 | 18 | 6 | 13.1 dB nominal | ~3.1-3.3 s | ~1.35 s | private events + saved IQ |

The shorter background benchmark used a faster capture/report setting. The proposal OTA run saved full IQ captures per report, increasing end-to-end time while preserving replay evidence.

## Limitations

- Latency depends strongly on capture size and whether IQ is saved for replay.
- The pipeline demonstrates event generation and deployed inference behavior; operational TAK/CoT emission is a separate integration artifact.
- TensorRT results are tied to the Jetson engine build and driver/runtime environment used during testing.

## Artifacts

- `results/cards/02_jetson_end_to_end_json_events.md`
- Private archive: gateway runner, raw event logs, replay manifests, and saved IQ.
