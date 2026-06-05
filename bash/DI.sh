#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${1:-config/DI.yaml}"
if [ "$#" -gt 0 ]; then
  shift
fi

python -m src.DI --config "${CONFIG_PATH}" "$@"
