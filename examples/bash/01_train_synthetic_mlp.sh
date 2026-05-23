#!/usr/bin/env bash
set -euo pipefail
# Canonical example script for refactored PSD/SNN CLI.
# Allowed representative: mean|median|element_psd|pca.
# Forbidden: raw trace CSV, same_label, legacy version keys.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/.example_out}"
RUN_ID="${RUN_ID:-example_run}"
mkdir -p "${OUTPUT_ROOT}"
python -m psd_snn.cli.train \
  --config "${PROJECT_ROOT}/examples/configs/runnable/train_synthetic_mlp_lif.json" \
  --output_dir "${OUTPUT_ROOT}/train_mlp" --epochs 1 --synthetic --run_id "${RUN_ID}_train"
