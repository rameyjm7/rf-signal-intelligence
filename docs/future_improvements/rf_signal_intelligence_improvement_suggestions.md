# RF Signal Intelligence Future Improvements

Repository: `rf-signal-intelligence`

This document tracks remaining improvements that have not yet been fully implemented. Completed items have been removed so this stays useful as a working roadmap.

## 1. Document Evaluation Protocols

**Goal:** Make reported metrics easier to interpret and reduce ambiguity around which results are directly comparable.

Create:

```text
docs/evaluation_protocols.md
```

Document each evaluation protocol separately:

```text
Protocol A: Full-distribution all-SNR test
Protocol B: High-SNR-only test
Protocol C: Class-balanced high-SNR slice
Protocol D: Cross-dataset ensemble sample
Protocol E: Noisy Drone held-out natural split
Protocol F: Noisy Drone held-out balanced split
```

For each protocol, include:

```text
dataset
sample count
SNR range
class balance
train/test split
checkpoint used
label mapping rule
metric
whether comparable to other rows
```

## 2. Add Reproducible Benchmark Tables

**Goal:** Make performance claims easy to rerun and compare across machines.

Add a small benchmark report that captures:

```text
model
runtime
hardware
batch size
input shape
precision
mean latency
p95 latency
throughput
command used
date run
```

Useful outputs:

```text
results/benchmarks/noisy_drone_onnx_cpu.md
results/benchmarks/noisy_drone_tensorrt_jetson.md
```

Minimum benchmark table:

| Runtime | Platform | Precision | Batch | Avg Latency | P95 Latency | Throughput |
|---|---|---|---:|---:|---:|---:|
| ONNX Runtime CPU | x86 or Jetson | FP32 | 1 | TBD | TBD | TBD |
| ONNX Runtime CUDA | Jetson | FP32 | 1 | TBD | TBD | TBD |
| TensorRT | Jetson | FP16 | 1 | TBD | TBD | TBD |
| TensorRT | Jetson | FP16 | 8 | TBD | TBD | TBD |

## 3. Add Runtime Profiling Notes

**Goal:** Capture where time is spent once the edge inference path is running.

Profile goals:

```text
end-to-end inference latency
CPU preprocessing time
accelerator inference time
host/device copy overhead
service/request overhead, if serving through an API
batching behavior
```

Expected documentation:

```text
docs/runtime_profile.md
```

Include:

```text
command used to capture the profile
hardware/software versions
screenshots or exported summaries if useful
bottlenecks found
changes made after profiling
```

## 4. Add An Optional RF Serving Example

**Goal:** Make the exported RF model easy to exercise from a small service or another repository without mixing service code into the research workflow.

Useful example files:

```text
examples/rf_signal_intelligence/
  README.md
  labels.json
  sample_input.npy
  curl_predict.sh
  benchmark_rf.py
```

The serving example should support shaped RF spectrogram inputs and use the same `labels.json` produced by `rfsi export-onnx`.

## 5. Expand OTA Test Coverage

**Goal:** Strengthen the live over-the-air result beyond the initial class sweep.

Useful follow-up runs:

```text
include Noise in class-sweep summaries
run multiple SNR thresholds, such as 20, 10, and 0 dB
repeat one lower-power TX setting
record clipping percentage and capture power in every summary table
save representative wins and misses, not every plot
```

Keep the language careful: these tests show controlled SDR replay/receive classification, not field collection from live drones.

## 6. Split The Live Classifier Into Smaller Modules

**Goal:** Keep the live SDR demo powerful while making the implementation easier to review, test, and extend.

Suggested package shape:

```text
src/rf_signal_intelligence/live/
  cli.py
  sdr_rx.py
  sdr_tx.py
  preprocessing.py
  inference.py
  sweep.py
  reporting.py
```

The current script can remain as the user-facing entrypoint, but most implementation details should move behind these modules. That would make the live RF classifier look more maintainable and production-style without changing the command-line workflow.
