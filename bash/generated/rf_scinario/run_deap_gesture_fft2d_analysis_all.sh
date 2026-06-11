#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
DATASETS=(deap dvs_gesture_flatten)
for ds in "${DATASETS[@]}"; do
  dir="$CONFIG_ROOT/fft2d_analysis/$ds"
  [[ -d "$dir" ]] || continue
  while IFS= read -r cfg; do
    echo "[fft2d_analysis] $cfg"
    bash "$ROOT/bash/fft2d_analysis.sh" "$cfg"
  done < <(find "$dir" -maxdepth 1 -type f -name '*.yaml' | LC_ALL=C sort)
done
