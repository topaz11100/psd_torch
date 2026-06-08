#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
export LOG_ROOT="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
mkdir -p "$LOG_ROOT"

MAX_PARALLEL="${MAX_PARALLEL:-1}"
mapfile -t CONFIGS < <(find "$CONFIG_ROOT/model_training" -type f -name '*.yaml' | sort)
mkdir -p "$LOG_ROOT/training_queue"
run_one() {
  local cfg="$1"
  local rel="${cfg#$CONFIG_ROOT/model_training/}"
  local stem="${rel%.yaml}"
  local safe="${stem//\//__}"
  local log="$LOG_ROOT/training_queue/${safe}.log"
  echo "[model_training] start $cfg -> $log"
  bash "$ROOT/bash/model_training.sh" "$cfg" > >(tee -a "$log") 2>&1
}
for cfg in "${CONFIGS[@]}"; do
  run_one "$cfg" &
  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
done
wait
