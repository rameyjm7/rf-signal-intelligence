# Result Cards

This folder contains one-page result cards for proposal/reviewer use. Each card follows the same structure: objective, setup/hardware, method, metrics, limitations, and artifacts.

| Card | Purpose |
|---|---|
| [01_background_false_positive.md](01_background_false_positive.md) | Background/no-drone rejection across Wi-Fi-heavy, Bluetooth-heavy, random 2.4 GHz, and soak conditions. |
| [02_jetson_end_to_end_json_events.md](02_jetson_end_to_end_json_events.md) | Jetson `sdr-gateway` IQ to preprocessing, TensorRT inference, JSONL event output, and measured latency. |
| [03_ota_futaba14_positive_control.md](03_ota_futaba14_positive_control.md) | Single-class OTA file-replay positive control with saved IQ replay evidence. |
| [04_multiclass_ota_positive_control.md](04_multiclass_ota_positive_control.md) | Multi-class OTA file-replay proposal table covering all six non-noise NoisyDroneRF classes plus no-TX controls. |
