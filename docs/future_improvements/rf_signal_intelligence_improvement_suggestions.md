# RF Signal Intelligence Repo Improvement Suggestions

Repository: `rf-signal-intelligence`  
Focus: Make the project stronger for NVIDIA, Jetson, TensorRT, Holoscan-adjacent sensor processing, and embedded AI roles.

## 1. Add an ONNX → TensorRT → Jetson Deployment Path

**Goal:** Turn the repo from an RFML training/evaluation project into a deployable NVIDIA edge-inference project.

Add a clean export and deployment path:

```text
exports/
  export_noisy_drone_to_onnx.py
  validate_onnx.py

deploy/
  build_tensorrt_engine.sh
  run_trtexec_benchmark.sh
  run_jetson_inference.py
  nsight_profile.sh

docs/
  jetson_tensorrt_deployment.md
```

Recommended target model: the best Noisy Drone RF classifier, because it has the strongest, cleanest headline result and is easiest to explain.

Suggested pipeline:

```text
Keras/PyTorch model
  -> ONNX export
  -> ONNX validation
  -> TensorRT FP16 engine
  -> Jetson inference
  -> trtexec benchmark
  -> Nsight Systems profile
```

Expected resume value:

```text
Exported RF/IQ deep-learning model to ONNX, converted it to TensorRT FP16 on NVIDIA Jetson, and benchmarked/profiled latency, throughput, and CPU/GPU bottlenecks with trtexec and Nsight Systems.
```

---

## 2. Make Evaluation Protocols Cleaner and Easier to Trust

**Goal:** Prevent reviewers from seeing the reported metrics as apples-to-oranges or cherry-picked.

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

This makes the repo look more rigorous and reduces ambiguity around why some scores are higher than others.

---

## 3. Convert the Best Notebooks into Reproducible CLI Workflows

**Goal:** Make the repo feel like a maintained ML engineering project instead of a notebook-driven research workspace.

Add or standardize commands like:

```bash
rfsi train --config configs/noisy_drone_vgg.yaml
rfsi evaluate --config configs/noisy_drone_vgg.yaml --checkpoint models/...best.keras
rfsi compare --config configs/evaluation_comparison.yaml
rfsi export-onnx --config configs/noisy_drone_vgg.yaml
```

Suggested README flow:

```text
1. Install
2. Download or point to dataset
3. Train or evaluate
4. Export artifacts
5. Reproduce headline table
6. Run live classifier
7. Deploy to TensorRT / Jetson
```

This makes the project easier for recruiters, hiring managers, and engineers to run without opening notebooks.

---

## 4. Promote the Live RF Classifier as a First-Class Demo

**Goal:** Since `scripts/live_noisy_drone_rf_classifier.py` already supports live playback/receive, make it obvious and reproducible from the top-level README.

Add a README section:

```markdown
## Live RF Drone Classifier

Run a live noisy-drone RF classifier using IQ playback or live SDR receive.

Pipeline:
IQ source -> windowing -> preprocessing/spectrogram -> model inference -> class/confidence -> latency/throughput reporting
```

Add example commands using the actual arguments from the script, for example:

```bash
python scripts/live_noisy_drone_rf_classifier.py \
  --source playback \
  --input data/sample_iq.pt \
  --model models/noisy_drone_rf_v2/best.keras \
  --window-size 1024 \
  --hop-size 512
```

If live SDR receive is supported, also include:

```bash
python scripts/live_noisy_drone_rf_classifier.py \
  --source sdr \
  --sample-rate 20e6 \
  --center-freq 2.437e9 \
  --model models/noisy_drone_rf_v2/best.keras
```

Suggested resume bullet:

```text
Built a live RF drone-classification pipeline with IQ playback/receive, windowed preprocessing, spectrogram-based deep-learning inference, confidence reporting, and latency instrumentation for real-time RF sensor-processing workflows.
```

---

## 5. Add Model Cards and Dataset Cards

**Goal:** Make the repo more credible, especially for reviewers who want to understand what the model detects, how it was trained, and where it can fail.

Add:

```text
docs/model_cards/
  noisy_drone_rf_v2_vgg.md
  rml2016_cnn_transformer.md
  rml2018_lstm.md
  deepradar2022_cnn_transformer.md

docs/dataset_cards/
  noisy_drone_rf_v2.md
  rml2016.md
  rml2018.md
  deepradar2022.md
```

Each model card should include:

```text
intended use
input shape
preprocessing
classes
training split
evaluation protocol
headline metrics
known limitations
latency target
export/deployment status
```

Each dataset card should include:

```text
source
class list
SNR range
sample count
preprocessing assumptions
license / usage limitations
leakage risks
recommended evaluation split
```

This makes the repo look more mature and helps prevent confusion around datasets, splits, and reported metrics.

---

# Priority Order

Recommended order for maximum NVIDIA-facing value:

1. **Promote and benchmark the live RF classifier**
2. **Add ONNX → TensorRT → Jetson deployment**
3. **Add evaluation protocol documentation**
4. **Convert key notebooks into CLI workflows**
5. **Add model and dataset cards**

The strongest final story is:

```text
RF Signal Intelligence provides a reproducible RFML pipeline from dataset training and evaluation through live IQ playback/receive, real-time drone classification, ONNX export, TensorRT deployment on NVIDIA Jetson, and Nsight-based performance profiling.
```
