# Dataset Card: Noisy Drone RF Signal Classification v2

## Source

Public/preprocessed NoisyDroneRFv2 I/Q files stored locally as `IQdata_sample*_target*_snr*.pt`. The repo expects a local path such as `/data/rameyjm7/datasets/NoisyDroneRFv2` or a path configured in `configs/noisy_drone_vgg.yaml`.

## Class List

`DJI`, `FutabaT14`, `FutabaT7`, `Graupner`, `Noise`, `Taranis`, `Turnigy`.

The preferred mapping source is `class_stats.csv`. If it is unavailable, labels are assigned from sorted integer target ids parsed from filenames.

## SNR Range

Repo examples and tests use filename-encoded SNR values. Current canonical evaluation filters to `SNR >= -6 dB`; live OTA replay reports commonly use `--tx-min-snr 20`.

## Sample Count

The dataset file count depends on the local dataset directory. Current committed evaluation artifacts report:

- 649 natural held-out test samples after `min_snr_db=-6`, `data_fraction=0.25`, and `test_size=0.20`.
- 203 balanced held-out test samples, 29 per class.
- Public aggregate OTA evidence is summarized in `results/cards/README.md`; detailed replay artifacts are private.

## Preprocessing Assumptions

- `.pt` payloads contain I/Q-like tensors or arrays.
- Samples are coerced to `(samples, 2)` float32.
- A high-power burst window is selected before spectrogram conversion.
- Full-complex STFT uses real and imaginary FFT channels, not log magnitude only.

## License / Usage Limitations

Verify the upstream dataset license before redistribution or commercial use. This repo documents workflows and local paths but does not make the dataset license broader than the source terms.

## Leakage Risks

- Avoid splitting multiple windows from the same source capture across train/test.
- Preserve stratified splits by parsed target id.
- Do not compare OTA replay results directly against offline held-out metrics.
- Keep `Noise` as a normal class unless the evaluation protocol explicitly excludes it.

## Recommended Evaluation Split

Use the canonical config:

- `min_snr_db=-6`
- `data_fraction=0.25`
- stratified `test_size=0.20`
- random seed `1961`

Report both natural held-out and balanced held-out metrics.
