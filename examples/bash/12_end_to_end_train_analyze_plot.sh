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
"${PROJECT_ROOT}/examples/bash/01_train_synthetic_mlp.sh"
"${PROJECT_ROOT}/examples/bash/03_analyze_signal_mean_median.sh"
"${PROJECT_ROOT}/examples/bash/07_analyze_fft2d_exact.sh"
python -m psd_snn.cli.analyze_signal --help >/dev/null
