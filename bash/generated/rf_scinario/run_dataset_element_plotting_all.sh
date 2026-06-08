#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
export LOG_ROOT="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
mkdir -p "$LOG_ROOT"

mapfile -t CONFIGS < <(find "$CONFIG_ROOT/plotting/dataset_signal" -type f \( -name 'dataset_element_psd.yaml' -o -name 'dataset_element_fft.yaml' \) | sort)
for cfg in "${CONFIGS[@]}"; do
  echo "[plotting] $cfg"
  bash "$ROOT/bash/plotting.sh" "$cfg"
done
