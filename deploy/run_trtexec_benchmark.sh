#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${1:-models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
DURATION="${DURATION:-30}"
WARMUP="${WARMUP:-500}"
ITERATIONS="${ITERATIONS:-1000}"

if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" || ! -x "$TRTEXEC" ]]; then
  echo "Could not find trtexec. Set TRTEXEC=/path/to/trtexec." >&2
  exit 1
fi
if [[ ! -f "$ENGINE_PATH" ]]; then
  echo "Missing TensorRT engine: $ENGINE_PATH" >&2
  exit 1
fi

"$TRTEXEC" \
  --loadEngine="$ENGINE_PATH" \
  --warmUp="$WARMUP" \
  --duration="$DURATION" \
  --iterations="$ITERATIONS"
