#!/usr/bin/env bash
# model_training 단계를 백그라운드로 실행하고 config별 로그 파일을 자동 생성한다.
set -euo pipefail

# 기본 실행 대상. 여러 시나리오를 고정해 둘 때는 이 배열에 JSON 경로를 추가한다.
# CLI 인자를 넘기면 아래 배열 대신 인자 목록을 실행한다.
CONFIG_PATHS=(
    "config/model_training.json"
)

if [ "$#" -gt 0 ]; then
    CONFIG_PATHS=( "$@" )
fi

STAGE="model_training"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${STAGE}"
mkdir -p "$LOG_DIR"

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    if [ -z "${CONFIG_PATH}" ]; then
        continue
    fi

    CONFIG_NAME="$(basename "$CONFIG_PATH")"
    CONFIG_STEM="${CONFIG_NAME%.json}"
    LOG_PATH="${LOG_DIR}/${RUN_STAMP}__${CONFIG_STEM}.log"

    nohup python src/model_training.py --config "$CONFIG_PATH" > "$LOG_PATH" 2>&1 &
    PID="$!"

    echo "STAGE=${STAGE}"
    echo "CONFIG=${CONFIG_PATH}"
    echo "LOG=${LOG_PATH}"
    echo "PID=${PID}"
done
