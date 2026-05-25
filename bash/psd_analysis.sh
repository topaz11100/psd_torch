#!/usr/bin/env bash
# psd_analysis 단계를 백그라운드로 실행하고 로그 파일을 자동 생성한다.
set -euo pipefail
CONFIG_PATH="${1:-config/psd_analysis.json}"
STAGE="psd_analysis"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_STEM="${CONFIG_NAME%.json}"
LOG_DIR="logs/${STAGE}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/${RUN_STAMP}__${CONFIG_STEM}.log"
nohup python src/psd_analysis.py --config "$CONFIG_PATH" > "$LOG_PATH" 2>&1 &
PID="$!"
echo "STAGE=${STAGE}"
echo "CONFIG=${CONFIG_PATH}"
echo "LOG=${LOG_PATH}"
echo "PID=${PID}"
