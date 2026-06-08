#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
export LOG_ROOT="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
mkdir -p "$LOG_ROOT"

mapfile -t CONFIGS < <(find "$CONFIG_ROOT/DI" -type f -name '*.yaml' | sort)
for cfg in "${CONFIGS[@]}"; do
  echo "[DI] $cfg"
  bash "$ROOT/bash/DI.sh" "$cfg"
done
