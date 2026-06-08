#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
export LOG_ROOT="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
mkdir -p "$LOG_ROOT"

mapfile -t CONFIGS < <(find "$CONFIG_ROOT/data_prep" -maxdepth 1 -type f -name '*.yaml' | sort)
for cfg in "${CONFIGS[@]}"; do
  echo "[data_prep] $cfg"
  bash "$ROOT/bash/data_prep.sh" "$cfg"
done
