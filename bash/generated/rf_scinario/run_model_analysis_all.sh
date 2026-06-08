#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
export LOG_ROOT="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
mkdir -p "$LOG_ROOT"

for stage in psd_analysis element_psd element_fft; do
  mapfile -t CONFIGS < <(find "$CONFIG_ROOT/$stage" -type f -name '*.yaml' | sort)
  for cfg in "${CONFIGS[@]}"; do
    echo "[$stage] $cfg"
    bash "$ROOT/bash/${stage}.sh" "$cfg"
  done
done
