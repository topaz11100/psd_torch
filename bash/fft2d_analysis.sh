#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/config/fft2d_analysis.yaml}"
source "$ROOT/bash/_logging.sh"
setup_psd_stage_logging "$(basename "$0" .sh)" "$CONFIG"
shift || true
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m src.2d_fft_analysis --config "$CONFIG" "$@"
