#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bash "$ROOT/bash/generated/rf_scinario/run_deap_gesture_model_psd_fft2d_analysis_all.sh"
bash "$ROOT/bash/generated/rf_scinario/run_deap_gesture_psd_fft2d_plotting_all.sh"
