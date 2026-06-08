#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/config/dataset_element_psd.yaml}"
source "$ROOT/bash/_logging.sh"
setup_psd_stage_logging "$(basename "$0" .sh)" "$CONFIG"
shift || true
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m src.dataset_element_psd --config "$CONFIG" "$@"
