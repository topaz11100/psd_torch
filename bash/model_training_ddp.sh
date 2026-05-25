#!/usr/bin/env bash
# model_training_ddp 단계를 백그라운드로 실행하고 로그 파일을 자동 생성한다.
set -euo pipefail
CONFIG_PATH="${1:-config/model_training_ddp.json}"
STAGE="model_training_ddp"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_STEM="${CONFIG_NAME%.json}"
LOG_DIR="logs/${STAGE}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/${RUN_STAMP}__${CONFIG_STEM}.log"
nohup torchrun --standalone --nproc_per_node=2 src/model_training.py --config "$CONFIG_PATH" --ddp true > "$LOG_PATH" 2>&1 &
PID="$!"
echo "STAGE=${STAGE}"
echo "CONFIG=${CONFIG_PATH}"
echo "LOG=${LOG_PATH}"
echo "PID=${PID}"
