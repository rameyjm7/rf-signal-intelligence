#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/usr/lib/python3.10/dist-packages:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/jake/workspace/.venv/bin/python}"
"$PYTHON" deploy/run_tensorrt_engine_inference.py "$@"
