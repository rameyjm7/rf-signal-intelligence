# Jetson TensorRT Deployment

This page documents the first-class deployment path for the NoisyDroneRFv2 RF classifier.

## Target Model

Use the best NoisyDroneRFv2 VGG full-complex spectrogram checkpoint:

```text
models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras
```

Why this model:

- strongest and cleanest headline result in the repo
- documented held-out and live OTA SDR-to-SDR evidence
- simple 7-class label set
- consistent spectrogram preprocessing path

## Pipeline

```text
Keras model
  -> ONNX export
  -> ONNX validation
  -> TensorRT FP16 engine
  -> Jetson inference
  -> trtexec benchmark
  -> Nsight Systems profile
```

## Install Extras

For export and ONNX Runtime validation:

```bash
pip install -e ".[deploy,noisy-drone]"
```

On Jetson, install the JetPack-provided TensorRT and Nsight Systems packages. Prefer the JetPack-matched TensorRT version over arbitrary PyPI wheels.

## 1. Export To ONNX

```bash
python exports/export_noisy_drone_to_onnx.py \
  --config configs/noisy_drone_vgg.yaml \
  --keras-model models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras \
  --out models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample-out models/noisy_drone_rf_v2/sample_input.npy \
  --labels-out models/noisy_drone_rf_v2/labels.json
```

Outputs:

- `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx`
- `models/noisy_drone_rf_v2/sample_input.npy`
- `models/noisy_drone_rf_v2/labels.json`

Pass `--sample-iq path/to/IQdata_sample...pt` to save a real spectrogram sample instead of a zero-valued shape sample.

## 2. Validate ONNX Against Keras

```bash
python exports/validate_onnx.py \
  --keras-model models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras \
  --onnx models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample models/noisy_drone_rf_v2/sample_input.npy \
  --labels models/noisy_drone_rf_v2/labels.json
```

Validation reports:

- Keras top-1 class and confidence
- ONNX top-1 class and confidence
- top-1 agreement
- max and mean absolute output error
- Keras latency
- ONNX Runtime latency
- selected ONNX Runtime providers

## 3. Build TensorRT FP16 Engine On Jetson

Run this on the Jetson so the engine is built for the target GPU and TensorRT version:

```bash
bash deploy/build_tensorrt_engine.sh \
  models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine
```

The script uses `/usr/src/tensorrt/bin/trtexec` when present, or `trtexec` from `PATH`.

## 4. Benchmark With trtexec

```bash
bash deploy/run_trtexec_benchmark.sh \
  models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine
```

Useful environment overrides:

```bash
DURATION=60 WARMUP=1000 ITERATIONS=2000 bash deploy/run_trtexec_benchmark.sh ...
```

Record benchmark output in this table when run on target hardware:

| Runtime | Platform | Precision | Batch | Avg Latency | P95 Latency | Throughput |
|---|---|---|---:|---:|---:|---:|
| ONNX Runtime CPU | Pending target run | FP32 | 1 | Pending | Pending | Pending |
| ONNX Runtime CUDA | Jetson | FP32 | 1 | Pending | Pending | Pending |
| ONNX Runtime TensorRT EP | Jetson | FP16 | 1 | Pending | Pending | Pending |
| TensorRT `trtexec` | Jetson | FP16 | 1 | Pending | Pending | Pending |

## 5. Run Jetson Inference

The Python runner uses ONNX Runtime provider priority:

```bash
python deploy/run_jetson_inference.py \
  --onnx models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample models/noisy_drone_rf_v2/sample_input.npy \
  --labels models/noisy_drone_rf_v2/labels.json \
  --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --iterations 100
```

The output includes prediction, confidence, selected providers, average latency, and throughput.

## 6. Profile With Nsight Systems

```bash
bash deploy/nsight_profile.sh \
  --onnx models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample models/noisy_drone_rf_v2/sample_input.npy \
  --labels models/noisy_drone_rf_v2/labels.json \
  --iterations 200
```

Profile goals:

- end-to-end inference latency
- CPU preprocessing time
- GPU inference time
- host-to-device and device-to-host copy overhead
- batching behavior

## Definition Of Done

- ONNX export script produces `.onnx`, `labels.json`, and `sample_input.npy`.
- ONNX validation agrees with Keras top-1 on the saved sample.
- Jetson builds a TensorRT FP16 engine with `trtexec`.
- Jetson benchmark numbers are recorded.
- Nsight Systems profile is captured and summarized.

## Resume Bullet

```text
Exported RF/IQ deep-learning model to ONNX, converted it to TensorRT FP16 on NVIDIA Jetson, and benchmarked/profiled latency, throughput, and CPU/GPU bottlenecks with trtexec and Nsight Systems.
```
