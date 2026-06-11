#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_ROOT="$ROOT/config/generated/rf_scinario"
DATASETS=(deap dvs_gesture_flatten)
for ds in "${DATASETS[@]}"; do
  for stage in plotting/psd_analysis plotting/fft2d_analysis; do
    dir="$CONFIG_ROOT/$stage/$ds"
    [[ -d "$dir" ]] || continue
    while IFS= read -r cfg; do
      echo "[plotting] $cfg"
      bash "$ROOT/bash/plotting.sh" "$cfg"
    done < <(find "$dir" -maxdepth 1 -type f -name '*.yaml' | LC_ALL=C sort)
  done
done
