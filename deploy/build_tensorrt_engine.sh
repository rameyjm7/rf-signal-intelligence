#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram.onnx}"
ENGINE_PATH="${2:-models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" || ! -x "$TRTEXEC" ]]; then
  echo "Could not find trtexec. Set TRTEXEC=/path/to/trtexec." >&2
  exit 1
fi
if [[ ! -f "$ONNX_PATH" ]]; then
  echo "Missing ONNX model: $ONNX_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"
"$TRTEXEC" \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --verbose

echo "Wrote TensorRT engine: $ENGINE_PATH"
