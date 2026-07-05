# NoisyDroneRFv2 TensorRT Jetson Benchmark

## Target

| Field | Value |
|---|---|
| Model | `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras` |
| ONNX | `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx` |
| TensorRT engine | `noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine` |
| Device | Jetson Orin |
| TensorRT | 10.3.0 |
| Nsight Systems | 2024.5.4 |
| Input shape | `1x1024x1024x2` |
| Output shape | `1x7` |

## TensorRT Benchmark

Command:

```bash
sudo env DURATION=5 WARMUP=100 ITERATIONS=100 ./benchmark_engine.sh
```

Result:

| Runtime | Precision | Batch | Mean Latency | P95 Latency | Throughput |
|---|---|---:|---:|---:|---:|
| TensorRT `trtexec` | FP16 | 1 | 79.0 ms | 79.01 ms | 12.58 qps |

## Runtime Baseline Table

| Runtime | Platform | Precision | Mean Latency | Status |
|---|---|---|---:|---|
| TensorRT `trtexec` | Jetson Orin | FP16 | 79.0 ms | Measured |

Planned baselines: Keras/TensorFlow FP32 and ONNX Runtime CPU/CUDA.

## TensorRT Correctness Check

Validated direct TensorRT engine inference on one preprocessed NoisyDroneRFv2 sample per class:

| Expected | TensorRT Prediction | Confidence |
|---|---|---:|
| DJI | DJI | 1.000 |
| FutabaT14 | FutabaT14 | 1.000 |
| FutabaT7 | FutabaT7 | 1.000 |
| Graupner | Graupner | 0.998 |
| Noise | Noise | 0.995 |
| Taranis | Taranis | 0.945 |
| Turnigy | Turnigy | 1.000 |

Summary: 7/7 expected classes matched.

## Nsight Systems Profile

Install command used on the Jetson:

```bash
sudo apt-get update
sudo apt-get install -y nsight-systems-2024.5.4
```

Capture command:

```bash
sudo OUT=outputs/nsight/noisy_drone_tensorrt \
  ITERATIONS=20 \
  bash deploy/nsight_profile.sh
```

Generated profile on the Jetson:

```text
~/rf-signal-intelligence-deploy/outputs/nsight/noisy_drone_tensorrt.nsys-rep
```

Inference output during profile:

```text
prediction : Noise (0.995)
top classes: Noise=0.995, FutabaT14=0.003, DJI=0.002
latency    : 92.09 ms
throughput : 10.86 infer/sec
```

Selected `nsys stats` observations:

| Area | Observation |
|---|---|
| NVTX | `TensorRT:ExecutionContext::enqueue` appeared 20 times, average 3.92 ms in the NVTX summary. |
| CUDA API | `cuMemcpyDtoHAsync_v2` dominated API time in the captured run, with 20 calls averaging 82.26 ms. |
| CUDA GPU kernels | Largest kernel family was FP16 scale kernels, 220 instances totaling 764.6 ms. |
| GPU memory ops | Host-to-device copies totaled 171.5 MB across 22 copies; device-to-host output copies were negligible. |

Notes:

- The direct TensorRT runner currently copies the full `1x1024x1024x2` float32 input for each inference.
- The small output tensor makes device-to-host transfer negligible.
- Further optimization should focus on reducing host-to-device transfer overhead, batching, or keeping preprocessed tensors resident on device when the upstream pipeline allows it.
