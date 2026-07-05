#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-/home/jake/workspace/SDR/rf-signal-intelligence/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
"$PYTHON" "/home/jake/workspace/SDR/rf-signal-intelligence/exports/run_onnx_inference.py" \
  --onnx "/home/jake/workspace/SDR/rf-signal-intelligence/models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx" \
  --input "/home/jake/workspace/SDR/rf-signal-intelligence/models/noisy_drone_rf_v2/sample_input.npy" \
  --labels "/home/jake/workspace/SDR/rf-signal-intelligence/models/noisy_drone_rf_v2/labels.json" \
  "$@"
