#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-/home/jake/workspace/SDR/rf-signal-intelligence/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
ONNX_MODEL="${ONNX_MODEL:?Set ONNX_MODEL to a local/private .onnx artifact}"
SAMPLE_INPUT="${SAMPLE_INPUT:?Set SAMPLE_INPUT to a local/private sample .npy artifact}"
"$PYTHON" "/home/jake/workspace/SDR/rf-signal-intelligence/exports/run_onnx_inference.py" \
  --onnx "$ONNX_MODEL" \
  --input "$SAMPLE_INPUT" \
  --labels "/home/jake/workspace/SDR/rf-signal-intelligence/models/noisy_drone_rf_v2/labels.json" \
  "$@"
