#!/usr/bin/env bash
# plotting 단계 실행 스크립트.
# CONFIG_GROUP_*는 2차원 배열 계약을 표현한다.
# - 같은 CONFIG_GROUP 안의 config들은 병렬 실행한다.
# - CONFIG_GROUPS에 나열된 1차 그룹은 앞 그룹이 모두 끝난 뒤 직렬 실행한다.
set -euo pipefail

CONFIG_GROUP_0=(
    "config/plotting.yaml"
)
CONFIG_GROUPS=(CONFIG_GROUP_0)

if [ "$#" -gt 0 ]; then
    # CLI 인자는 하나의 병렬 그룹으로 취급한다.
    CONFIG_GROUP_0=( "$@" )
    CONFIG_GROUPS=(CONFIG_GROUP_0)
fi

STAGE="plotting"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${STAGE}"
mkdir -p "$LOG_DIR"
LAST_PID=""

launch_config() {
    local CONFIG_PATH="$1"
    local GROUP_INDEX="$2"
    local CONFIG_NAME CONFIG_STEM LOG_PATH PID
    CONFIG_NAME="$(basename "$CONFIG_PATH")"
    CONFIG_STEM="${CONFIG_NAME%.*}"
    LOG_PATH="${LOG_DIR}/${RUN_STAMP}__g${GROUP_INDEX}__${CONFIG_STEM}.log"

    CONFIG_PATH="$CONFIG_PATH" nohup bash -c 'python src/plotting.py --config "$CONFIG_PATH"' > "$LOG_PATH" 2>&1 &
    PID="$!"
    LAST_PID="$PID"
    echo "STAGE=${STAGE}"
    echo "GROUP=${GROUP_INDEX}"
    echo "CONFIG=${CONFIG_PATH}"
    echo "LOG=${LOG_PATH}"
    echo "PID=${PID}"
    echo
}

GROUP_INDEX=0
for GROUP_NAME in "${CONFIG_GROUPS[@]}"; do
    declare -n GROUP="$GROUP_NAME"
    PIDS=()
    echo "[${STAGE}] group ${GROUP_INDEX} start: ${#GROUP[@]} config(s)"
    for CONFIG_PATH in "${GROUP[@]}"; do
        if [ -z "$CONFIG_PATH" ]; then
            continue
        fi
        launch_config "$CONFIG_PATH" "$GROUP_INDEX"
        PIDS+=("$LAST_PID")
    done

    GROUP_STATUS=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            GROUP_STATUS=1
        fi
    done
    if [ "$GROUP_STATUS" -ne 0 ]; then
        echo "[${STAGE}] group ${GROUP_INDEX} failed" >&2
        exit "$GROUP_STATUS"
    fi
    echo "[${STAGE}] group ${GROUP_INDEX} done"
    GROUP_INDEX=$((GROUP_INDEX + 1))
done
