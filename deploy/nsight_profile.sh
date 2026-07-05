#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-outputs/nsight/noisy_drone_rf_inference}"
NSYS="${NSYS:-nsys}"
ENGINE="${ENGINE:-models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_fp16.engine}"
SAMPLE="${SAMPLE:-models/noisy_drone_rf_v2/sample_input.npy}"
LABELS="${LABELS:-models/noisy_drone_rf_v2/labels.json}"
ITERATIONS="${ITERATIONS:-20}"

if ! command -v "$NSYS" >/dev/null 2>&1; then
  if [[ "$NSYS" == "nsys" && -x /usr/lib/nsight-systems/bin/nsys ]]; then
    NSYS="/usr/lib/nsight-systems/bin/nsys"
  elif [[ "$NSYS" == "nsys" && -x /usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-armv8/nsys ]]; then
    NSYS="/usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-armv8/nsys"
  fi
fi

if ! command -v "$NSYS" >/dev/null 2>&1; then
  echo "Could not find nsys. Install Nsight Systems or set NSYS=/path/to/nsys." >&2
  exit 1
fi

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run this with sudo on Jetson so TensorRT can initialize CUDA." >&2
  echo "Example: sudo OUT=$OUT ITERATIONS=$ITERATIONS bash deploy/nsight_profile.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  --output="$OUT" \
  ./deploy/run_tensorrt_engine_inference.sh \
    --engine "$ENGINE" \
    --sample "$SAMPLE" \
    --labels "$LABELS" \
    --iterations "$ITERATIONS" \
    --format table \
    "$@"

echo "Wrote Nsight Systems profile prefix: $OUT"
