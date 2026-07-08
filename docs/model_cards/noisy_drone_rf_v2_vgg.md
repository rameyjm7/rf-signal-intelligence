# Model Card: NoisyDroneRFv2 VGG Full-Complex Spectrogram

## Summary

| Field | Value |
|---|---|
| Model id | `noisy_drone_rf_v2_vgg_full_complex_spectrogram` |
| Checkpoint | `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras` |
| Dataset | Noisy Drone RF Signal Classification v2 |
| Architecture | VGG-style 2D CNN over full-complex STFT spectrograms |
| Task | 7-way drone/controller/noise RF classification |
| Status | Canonical NoisyDroneRFv2 model for this repo |

## Intended Use

Classify labeled NoisyDroneRFv2-style I/Q captures and demonstrate RFML preprocessing, export, and deployment workflows with user-supplied data and models.

This model is appropriate for research demos, reproducible evaluation, and feasibility evidence for RFML classification. It is not a field-validated drone detector and should not be presented as operational coverage against arbitrary real-world emitters.

## Input Shape

`(1024, 1024, 2)` float32 tensor.

The two channels are the real and imaginary parts of the complex STFT after FFT shift, normalization, and clipping.

## Preprocessing

- Load `.pt` I/Q sample.
- Coerce to `(samples, 2)` float32 I/Q.
- Select or provide a fixed-size IQ window with `sample_len=1048576`.
- Normalize the I/Q window.
- Compute full-complex STFT with `nfft=1024`, `hop=1024`.
- Resize/pad time axis to `time_bins=1024`.
- Normalize spectrogram by standard deviation and clip to `[-6, 6]`.

Live SDR framing and gateway integration are maintained in the private product repository.

## Classes

`DJI`, `FutabaT14`, `FutabaT7`, `Graupner`, `Noise`, `Taranis`, `Turnigy`.

Label mapping is read from `class_stats.csv` when available and otherwise follows sorted integer target ids from NoisyDroneRFv2 filenames.

## Training Split

The canonical notebook/CLI split uses:

- `min_snr_db=-6`
- `data_fraction=0.25`
- stratified train/test split with `test_size=0.20`
- stratified validation split from the training partition
- random seed `1961`
- optional class-balanced replay for training

The config-driven workflow is `rfsi train --config configs/noisy_drone_vgg.yaml`.

## Evaluation Protocol

Primary repo protocol:

- Held-out natural split using the same filtered manifest.
- Balanced held-out slice sampled from the natural test split.
- One evaluation window per sample.

Related artifacts:

- `outputs/noisy_drone_rf_v2_eval/44_noisy_drone_rf_v2_vgg_full_complex_spectrogram_metrics.json`
- `outputs/50_evaluation_comparison/50_noisy_drone_rf_v2_eval_metrics.json`
- `results/cards/README.md`

## Headline Metrics

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| Natural held-out test | 0.9769 | 0.9775 | 0.9767 | 649 |
| Balanced held-out test | 0.9803 | 0.9807 | 0.9807 | 203 |
| Live OTA SNR >= 20 dB class sweep | See report | See report | See report | 70 trials |

The live OTA report uses SDR replay/receive and is not directly comparable to offline held-out dataset metrics.

## Known Limitations

- Trained and evaluated on public/preprocessed NoisyDroneRFv2 I/Q samples, not broad live field captures.
- High accuracy can depend on the dataset split, SNR filter, replay path, and confidence policy.
- OTA replay demonstrates hardware-path survivability but not arbitrary-range drone detection.
- Saturation/clipping in SDR capture can alter predictions; live reports include clipping diagnostics.
- Noise is a learned class in the dataset and does not cover all possible RF background conditions.

## Latency Target

The public ONNX/TensorRT helpers report inference latency and throughput. The practical target is interactive single-window classification on a workstation or accelerated edge device; hard real-time latency is not guaranteed until benchmarked on the target platform.

## Export / Deployment Status

- Keras checkpoint: available.
- ONNX export path: `rfsi export-onnx --config configs/noisy_drone_vgg.yaml`, which writes ONNX, `sample_input.npy`, `labels.json`, a runnable `run_onnx_inference.sh` helper, and validates ONNX Runtime output against Keras by default.
- First-class ONNX to TensorRT / Jetson deployment path: `docs/jetson_tensorrt_deployment.md`.
