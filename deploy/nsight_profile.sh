#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-outputs/nsight/noisy_drone_rf_inference}"
PYTHON="${PYTHON:-python3}"
NSYS="${NSYS:-nsys}"

if ! command -v "$NSYS" >/dev/null 2>&1; then
  echo "Could not find nsys. Install Nsight Systems or set NSYS=/path/to/nsys." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  --output="$OUT" \
  "$PYTHON" deploy/run_jetson_inference.py "$@"

echo "Wrote Nsight Systems profile prefix: $OUT"
