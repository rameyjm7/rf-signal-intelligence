# Jetson TensorRT RF Inference Plan

This note captures a future improvement path that connects the RF model work in this repository to an NVIDIA Jetson deployment story using ONNX, TensorRT, and Nsight Systems.

## Goal

Build a polished NVIDIA-facing demo:

```text
RF model training/evaluation
  -> ONNX export
  -> ONNX validation
  -> FastAPI inference service
  -> TensorRT engine on Jetson
  -> benchmark and profile
```

The target story:

> Trained and validated RF ML models in `rf-signal-intelligence`, exported a model to ONNX, deployed it through `gpu-inference-pipeline`, converted it to TensorRT on Jetson, benchmarked latency and throughput, and profiled the inference path with Nsight Systems.

## Repository Split

Use both repositories, with clear responsibilities.

| Repository | Role |
|---|---|
| `rf-signal-intelligence` | Model source, training/evaluation, ONNX export, ONNX validation, labels, model cards, sample inputs |
| `gpu-inference-pipeline` | Deployment/demo service, ONNX Runtime serving, benchmarking, TensorRT notes/scripts, Jetson/Nsight documentation |

This keeps the RF research artifacts separate from the production-style inference service.

## First Model To Export

Start with the NoisyDroneRFv2 VGG full-complex spectrogram model:

```text
models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras
```

Why this model:

- It has the cleanest headline metrics in the current repo.
- It already has live OTA SDR validation artifacts.
- It is easier to explain than models with label-order or dataset-calibration caveats.

## Phase 1: ONNX Export In `rf-signal-intelligence`

Add export and validation scripts:

```text
scripts/export_noisy_drone_to_onnx.py
scripts/validate_noisy_drone_onnx.py
```

Add model deployment artifacts:

```text
models/noisy_drone_rf_v2/noisy_drone_rf_v2.onnx
models/noisy_drone_rf_v2/labels.json
models/noisy_drone_rf_v2/sample_input.npy
models/noisy_drone_rf_v2/model_card.md
```

Expected export command:

```bash
python scripts/export_noisy_drone_to_onnx.py \
  --keras-model models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras \
  --out models/noisy_drone_rf_v2/noisy_drone_rf_v2.onnx \
  --sample-out models/noisy_drone_rf_v2/sample_input.npy \
  --labels-out models/noisy_drone_rf_v2/labels.json
```

Expected validation command:

```bash
python scripts/validate_noisy_drone_onnx.py \
  --keras-model models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras \
  --onnx models/noisy_drone_rf_v2/noisy_drone_rf_v2.onnx \
  --sample models/noisy_drone_rf_v2/sample_input.npy \
  --labels models/noisy_drone_rf_v2/labels.json
```

Validation should report:

- Keras top-1 class and confidence
- ONNX top-1 class and confidence
- top-1 agreement
- maximum absolute output error
- mean absolute output error
- Keras latency
- ONNX Runtime latency

## Phase 2: RF Example In `gpu-inference-pipeline`

Add an RF example directory:

```text
examples/rf_signal_intelligence/
  README.md
  labels.json
  sample_input.npy
  curl_predict.sh
  benchmark_rf.py
```

Add TensorRT and profiling docs/scripts:

```text
docs/tensorrt_jetson.md
docs/nsight_profile.md
scripts/build_trt_engine.sh
scripts/benchmark_trt.sh
```

The service should be able to run the exported RF model through ONNX Runtime:

```bash
export MODEL_PATH=examples/rf_signal_intelligence/noisy_drone_rf_v2.onnx
export ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"

uvicorn gpu_inference_pipeline.app:app --host 0.0.0.0 --port 8000
```

The RF example should support shaped RF/spectrogram inputs rather than only flat toy vectors.

## Phase 3: TensorRT On Jetson

Build a TensorRT FP16 engine on the Jetson:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=noisy_drone_rf_v2.onnx \
  --saveEngine=noisy_drone_rf_v2_fp16.engine \
  --fp16 \
  --verbose
```

Benchmark the TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=noisy_drone_rf_v2_fp16.engine \
  --warmUp=500 \
  --duration=30 \
  --iterations=1000
```

Capture results in `docs/tensorrt_jetson.md`.

Minimum benchmark table:

| Runtime | Platform | Precision | Batch | Avg Latency | P95 Latency | Throughput |
|---|---|---|---:|---:|---:|---:|
| ONNX Runtime CPU | x86 or Jetson | FP32 | 1 | TBD | TBD | TBD |
| ONNX Runtime CUDA | Jetson | FP32 | 1 | TBD | TBD | TBD |
| TensorRT | Jetson | FP16 | 1 | TBD | TBD | TBD |
| TensorRT | Jetson | FP16 | 8 | TBD | TBD | TBD |

## Phase 4: Nsight Systems Profiling

Use Nsight Systems after the Jetson inference path works.

Profile goals:

- end-to-end inference latency
- CPU preprocessing time
- GPU inference time
- host-to-device and device-to-host copy overhead
- service/request overhead
- batching behavior

Expected documentation:

```text
docs/nsight_profile.md
```

Include:

- command used to capture the profile
- hardware/software versions
- screenshots or exported summaries if appropriate
- bottlenecks found
- changes made after profiling

## Priority Order

Learn/build in this order:

1. ONNX export and validation
2. TensorRT engine build with `trtexec`
3. Jetson deployment
4. Nsight Systems profiling
5. CUDA Runtime basics
6. Custom CUDA kernels only if needed later

## Suggested Project Name

```text
Jetson TensorRT RF Signal Inference Pipeline
```

## Resume Bullet

> Exported a NoisyDroneRFv2 RF classifier to ONNX, deployed it through a FastAPI GPU inference service, converted it to a TensorRT FP16 engine on NVIDIA Jetson, and benchmarked/profiled latency using `trtexec` and Nsight Systems.

## Definition Of Done

The demo is ready to show when these are complete:

- ONNX export script exists and is repeatable.
- ONNX validation matches Keras top-1 output on a saved sample input.
- `gpu-inference-pipeline` can serve the ONNX model through `/predict`.
- Jetson can build a TensorRT FP16 engine from the ONNX model.
- Jetson benchmark numbers are recorded.
- Nsight Systems profile notes are documented.
- README/model-card text explains the path from RF dataset model to deployable NVIDIA inference pipeline.
