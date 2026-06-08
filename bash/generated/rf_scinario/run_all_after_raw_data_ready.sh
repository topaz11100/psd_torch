#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
export LOG_ROOT="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
mkdir -p "$LOG_ROOT"

bash "$ROOT/bash/generated/rf_scinario/run_data_prep_all.sh"
bash "$ROOT/bash/generated/rf_scinario/run_dataset_signal_all.sh"
bash "$ROOT/bash/generated/rf_scinario/run_training_queue.sh"
bash "$ROOT/bash/generated/rf_scinario/run_model_analysis_all.sh"
bash "$ROOT/bash/generated/rf_scinario/run_plotting_all.sh"
bash "$ROOT/bash/generated/rf_scinario/run_DI_all.sh"
