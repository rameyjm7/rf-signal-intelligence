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
rfsi export-onnx \
  --config configs/noisy_drone_vgg.yaml \
  --checkpoint models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras \
  --out models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample-out models/noisy_drone_rf_v2/sample_input.npy \
  --labels-out models/noisy_drone_rf_v2/labels.json
```

Outputs:

- `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx`
- `models/noisy_drone_rf_v2/sample_input.npy`
- `models/noisy_drone_rf_v2/labels.json`
- `models/noisy_drone_rf_v2/run_onnx_inference.sh`

Pass `--sample-iq path/to/IQdata_sample...pt` to save a real spectrogram sample instead of a zero-valued shape sample.

By default this command also validates ONNX Runtime output against the Keras checkpoint. Use `--no-validate` only when you are exporting on a machine without `onnxruntime` installed.

Immediately run a local CPU classification from the exported sample:

```bash
models/noisy_drone_rf_v2/run_onnx_inference.sh --providers CPUExecutionProvider
```

For demos where the raw top-1 is `Noise` but you want to inspect the strongest RF class, use:

```bash
models/noisy_drone_rf_v2/run_onnx_inference.sh \
  --providers CPUExecutionProvider \
  --decision-mode non-noise
```

To classify raw IQ instead of an already prepared spectrogram tensor, pass `--iq-file`. The script will scan candidate model windows across the capture, preprocess each candidate into a full-complex spectrogram, and select the best window by target-class or non-noise confidence:

```bash
models/noisy_drone_rf_v2/run_onnx_inference.sh \
  --iq-file outputs/rx_debug.npy \
  --target-class FutabaT14 \
  --window-score-mode target \
  --decision-mode non-noise \
  --providers CPUExecutionProvider
```

Run a compact per-class sanity check against mounted NoisyDroneRFv2 data:

```bash
models/noisy_drone_rf_v2/run_onnx_inference.sh \
  --class-sweep \
  --dataset-dir /data/rameyjm7/datasets/NoisyDroneRFv2 \
  --min-snr 20 \
  --samples-per-class 1 \
  --max-predictions 8 \
  --format table \
  --providers CPUExecutionProvider
```

## 2. Validate ONNX Against Keras

The export command validates by default. To run validation separately:

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
| TensorRT `trtexec` | Jetson Orin | FP16 | 1 | 79.0 ms | 79.01 ms | 12.58 qps |

Recorded TensorRT run:

```text
TensorRT: 10.3.0
Device: Jetson Orin
Input: 1x1024x1024x2
Output: 1x7
Command: sudo env DURATION=5 WARMUP=100 ITERATIONS=100 ./benchmark_engine.sh
```

Committed benchmark/profile summary: `results/benchmarks/noisy_drone_tensorrt_jetson.md`.

## 5. Run TensorRT Engine Inference

Use the direct TensorRT runner to verify that the built engine produces decoded class predictions, not just benchmark timings:

```bash
sudo ./deploy/run_tensorrt_engine_inference.sh \
  --engine models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine \
  --sample models/noisy_drone_rf_v2/sample_input.npy \
  --labels models/noisy_drone_rf_v2/labels.json \
  --iterations 10 \
  --format table
```

The output includes prediction, confidence, top classes, average latency, throughput, and TensorRT input/output tensor shapes.

Validated TensorRT FP16 correctness on the Jetson with one preprocessed NoisyDroneRFv2 sample per class:

| Expected | TensorRT Prediction | Confidence |
|---|---|---:|
| DJI | DJI | 1.000 |
| FutabaT14 | FutabaT14 | 1.000 |
| FutabaT7 | FutabaT7 | 1.000 |
| Graupner | Graupner | 0.998 |
| Noise | Noise | 0.995 |
| Taranis | Taranis | 0.945 |
| Turnigy | Turnigy | 1.000 |

Summary: 7/7 expected classes matched using direct TensorRT engine inference.

## 5.1 Run Live Gateway-to-TensorRT Event Demo

Use the gateway RX classifier when the Jetson should prove the complete live
path, not just a preprocessed TensorRT sample:

```bash
cd /home/jake/workspace/SDR/rf-signal-intelligence
/home/jake/workspace/SDR/SDR-Shark/backend/.venv/bin/python \
  scripts/noisy_drone_gateway_rx_classifier.py \
  --gateway-url http://127.0.0.1:8080 \
  --device-id bladerf:0 \
  --freq 2470000000 \
  --sample-rate 20000000 \
  --bandwidth 20000000 \
  --lna-gain 40 \
  --vga-gain 50 \
  --backend tensorrt \
  --engine models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine \
  --labels models/noisy_drone_rf_v2/labels.json \
  --capture-samples 4194304 \
  --discard-captures 1 \
  --window-score-mode raw \
  --decision-mode hybrid \
  --min-snr-db 5 \
  --event-jsonl results/benchmarks/jetson_noisy_drone_events.jsonl \
  --format event-json \
  --continuous \
  --interval-sec 1
```

This emits one compact JSON event per capture and appends the same event to the
JSONL artifact. The event includes the SDR source, final prediction, top
classes, quality gates, selected IQ window, and timing fields for capture read,
preprocessing, TensorRT inference, decision, quality/gates, classify total, and
end-to-end latency.

See `results/benchmarks/noisy_drone_jetson_end_to_end_demo.md` for the demo
checklist and event schema.

For a stronger correctness check, copy one or more preprocessed validation `.npy` files to the Jetson and run:

```bash
for cls in DJI FutabaT14 FutabaT7 Graupner Noise Taranis Turnigy; do
  echo "=== ${cls} ==="
  sudo ./deploy/run_tensorrt_engine_inference.sh \
    --sample "validation_samples/${cls}.npy" \
    --iterations 1 \
    --format table
done
```

If the NoisyDroneRFv2 `.pt` dataset and compatible PyTorch install are both available on the Jetson, the same runner can classify raw IQ samples or run a small class sweep:

```bash
sudo ./deploy/run_tensorrt_engine_inference.sh \
  --class-sweep \
  --dataset-dir /data/rameyjm7/datasets/NoisyDroneRFv2 \
  --samples-per-class 1 \
  --min-snr 20 \
  --format table
```

## 6. Optional ONNX Runtime Provider Inference

The ONNX Runtime runner uses provider priority:

```bash
python deploy/run_jetson_inference.py \
  --onnx models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx \
  --sample models/noisy_drone_rf_v2/sample_input.npy \
  --labels models/noisy_drone_rf_v2/labels.json \
  --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --iterations 100
```

This is useful when ONNX Runtime CUDA or TensorRT execution providers are installed. The direct TensorRT engine runner above is the primary Jetson correctness path.

## 7. Profile With Nsight Systems

Install Nsight Systems on the Jetson if `nsys` is not present. Prefer the NVIDIA Jetson repo package over Ubuntu's older `nsight-systems` package:

```bash
sudo apt-get update
sudo apt-get install -y nsight-systems-2024.5.4
```

The helper auto-detects `nsys` from `PATH` and common Jetson package paths. If your install is elsewhere, pass `NSYS=/path/to/nsys`.

Capture a TensorRT engine profile:

```bash
sudo OUT=outputs/nsight/noisy_drone_tensorrt \
  ITERATIONS=20 \
  bash deploy/nsight_profile.sh
```

To profile a different preprocessed validation sample:

```bash
sudo OUT=outputs/nsight/noisy_drone_tensorrt_dji \
  SAMPLE=validation_samples/DJI.npy \
  ITERATIONS=20 \
  bash deploy/nsight_profile.sh
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
