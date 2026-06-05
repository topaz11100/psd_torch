#!/usr/bin/env bash
# model_training_ddp 단계 실행 스크립트.
# CONFIG_GROUP_*는 2차원 배열 계약을 표현한다.
# - 같은 CONFIG_GROUP 안의 config들은 병렬 실행한다.
# - CONFIG_GROUPS에 나열된 1차 그룹은 앞 그룹이 모두 끝난 뒤 직렬 실행한다.
# - CLI 인자가 디렉터리이면 leaf 폴더별로 직렬화하고, 같은 폴더 안의 config들은 병렬화한다.
# - 단, d_rf를 제외한 RF neuron-family config들은 같은 leaf 폴더 안에서도 RF-only 병렬 그룹으로 분리한다.
#
# Compile-cache options:
#   --compile-cache-root PATH     cache root directory. Default: $PSD_TORCH_COMPILE_CACHE_ROOT or /home/yongokhan/workspace/cache/torch_compile
#   --experiment-name NAME        experiment namespace under the cache root. Default: $PSD_EXPERIMENT_NAME or RUN_STAMP
#   --compile-cache-dir PATH      exact cache experiment directory. Overrides root/name.
#   --no-config-cache-subdir      use the experiment directory directly instead of appending CONFIG_STEM.
#
# Example:
#   bash/model_training_ddp.sh --compile-cache-root /home/yongokhan/workspace/cache/torch_compile --experiment-name 04_exact_pca config/a.yaml
# This sets PSD_TORCH_COMPILE_CACHE_DIR=/tmp/psd_cache/04_exact_pca/a for that config;
# src/model_training.py then appends rank0/rank1 below that directory.
set -euo pipefail

CONFIG_GROUP_0=(
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/control/simple/shd_lif_soft_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/control/simple/s-mnist_lif_soft_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/membrane_constant_fixed/simple/s-mnist_lif_soft_alpha0p5_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/membrane_constant_fixed/simple/shd_lif_soft_alpha0p5_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/threshold_train/simple/s-mnist_lif_soft_train.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/threshold_train/simple/shd_lif_soft_train.yaml"
)

CONFIG_GROUP_1=(
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/control/simple/s-mnist_rf_none_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/control/simple/shd_rf_none_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/membrane_constant_fixed/simple/s-mnist_rf_none_fc0p25_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/membrane_constant_fixed/simple/shd_rf_none_fc0p25_fixed.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/threshold_train/simple/shd_rf_none_train.yaml"
    "/home/yongokhan/workspace/code/psd/config/neuron_characterization_scenarios/train/threshold_train/simple/s-mnist_rf_none_train.yaml"
)

CONFIG_GROUPS=(CONFIG_GROUP_0 CONFIG_GROUP_1)

STAGE="model_training_ddp"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/home/yongokhan/workspace/logs/${STAGE}"
COMPILE_CACHE_ROOT="${PSD_TORCH_COMPILE_CACHE_ROOT:-/home/yongokhan/workspace/cache/torch_compile}"
EXPERIMENT_NAME="${PSD_EXPERIMENT_NAME:-$RUN_STAMP}"
COMPILE_CACHE_DIR="${PSD_TORCH_COMPILE_CACHE_DIR_BASE:-}"
CONFIG_CACHE_SUBDIR="${PSD_TORCH_COMPILE_CACHE_PER_CONFIG:-true}"
CONFIG_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --compile-cache-root)
            if [ "$#" -lt 2 ]; then echo "--compile-cache-root requires PATH" >&2; exit 2; fi
            COMPILE_CACHE_ROOT="$2"; shift 2 ;;
        --compile-cache-root=*)
            COMPILE_CACHE_ROOT="${1#*=}"; shift ;;
        --experiment-name|--experiment)
            if [ "$#" -lt 2 ]; then echo "--experiment-name requires NAME" >&2; exit 2; fi
            EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*|--experiment=*)
            EXPERIMENT_NAME="${1#*=}"; shift ;;
        --compile-cache-dir)
            if [ "$#" -lt 2 ]; then echo "--compile-cache-dir requires PATH" >&2; exit 2; fi
            COMPILE_CACHE_DIR="$2"; shift 2 ;;
        --compile-cache-dir=*)
            COMPILE_CACHE_DIR="${1#*=}"; shift ;;
        --no-config-cache-subdir)
            CONFIG_CACHE_SUBDIR="false"; shift ;;
        --help|-h)
            sed -n '1,34p' "$0"
            exit 0 ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do CONFIG_ARGS+=("$1"); shift; done ;;
        --*)
            echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            CONFIG_ARGS+=("$1"); shift ;;
    esac
done

is_rf_non_drf_config() {
    local CONFIG_PATH="$1"
    python - "$CONFIG_PATH" <<'PYRF'
import sys
from pathlib import Path
from src.util.config import load_structured
p = Path(sys.argv[1])
data = load_structured(p)
if isinstance(data, dict) and isinstance(data.get('model_training'), dict):
    data = data['model_training']
nt = str(data.get('neuron_type', data.get('model', ''))).strip().lower().replace('-', '_')
model = str(data.get('model', '')).strip().lower().replace('-', '_')
is_rf = (nt == 'rf' or nt.endswith('_rf') or model.startswith('rf') or '_rf_' in model or model.startswith('vgg11_rf') or model.startswith('resnet18_rf'))
is_drf = (nt == 'd_rf' or model.startswith('d_rf'))
sys.exit(0 if (is_rf and not is_drf) else 1)
PYRF
}

add_config_group() {
    local GROUP_NAME="CONFIG_GROUP_${#CONFIG_GROUPS[@]}"
    local ASSIGN="$GROUP_NAME=("
    local ITEM QUOTED
    for ITEM in "$@"; do
        printf -v QUOTED '%q' "$ITEM"
        ASSIGN+=" $QUOTED"
    done
    ASSIGN+=" )"
    eval "$ASSIGN"
    CONFIG_GROUPS+=("$GROUP_NAME")
}

add_directory_groups() {
    local DIR="$1"
    local LEAF CONFIG_PATH
    mapfile -t LEAF_DIRS < <(find "$DIR" -type f \( -name '*.yaml' -o -name '*.yml' \) -printf '%h\n' | sort -u)
    for LEAF in "${LEAF_DIRS[@]}"; do
        mapfile -t FILES < <(find "$LEAF" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort)
        NON_RF=()
        RF_ONLY=()
        for CONFIG_PATH in "${FILES[@]}"; do
            if is_rf_non_drf_config "$CONFIG_PATH"; then
                RF_ONLY+=("$CONFIG_PATH")
            else
                NON_RF+=("$CONFIG_PATH")
            fi
        done
        if [ "${#NON_RF[@]}" -gt 0 ]; then add_config_group "${NON_RF[@]}"; fi
        if [ "${#RF_ONLY[@]}" -gt 0 ]; then add_config_group "${RF_ONLY[@]}"; fi
    done
}

if [ "${#CONFIG_ARGS[@]}" -gt 0 ]; then
    CONFIG_GROUPS=()
    DIRECT_FILES=()
    for ITEM in "${CONFIG_ARGS[@]}"; do
        if [ -d "$ITEM" ]; then
            if [ "${#DIRECT_FILES[@]}" -gt 0 ]; then
                add_config_group "${DIRECT_FILES[@]}"
                DIRECT_FILES=()
            fi
            add_directory_groups "$ITEM"
        else
            DIRECT_FILES+=("$ITEM")
        fi
    done
    if [ "${#DIRECT_FILES[@]}" -gt 0 ]; then
        add_config_group "${DIRECT_FILES[@]}"
    fi
fi

mkdir -p "$LOG_DIR"
LAST_PID=""

sanitize_path_component() {
    local VALUE="$1"
    VALUE="${VALUE//\\//_}"
    VALUE="${VALUE// /_}"
    VALUE="${VALUE//:/_}"
    printf '%s' "$VALUE"
}

compile_cache_base_for_config() {
    local CONFIG_STEM="$1"
    local BASE_DIR
    if [ -n "$COMPILE_CACHE_DIR" ]; then
        BASE_DIR="$COMPILE_CACHE_DIR"
    else
        BASE_DIR="${COMPILE_CACHE_ROOT%/}/$(sanitize_path_component "$EXPERIMENT_NAME")"
    fi
    case "$(printf '%s' "$CONFIG_CACHE_SUBDIR" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on)
            printf '%s/%s' "${BASE_DIR%/}" "$(sanitize_path_component "$CONFIG_STEM")" ;;
        *)
            printf '%s' "$BASE_DIR" ;;
    esac
}

launch_config() {
    local CONFIG_PATH="$1"
    local GROUP_INDEX="$2"
    local CONFIG_NAME CONFIG_STEM LOG_PATH PID CONFIG_CACHE_DIR_RESOLVED
    CONFIG_NAME="$(basename "$CONFIG_PATH")"
    CONFIG_STEM="${CONFIG_PATH%.*}"
    CONFIG_STEM="${CONFIG_STEM#./}"
    CONFIG_STEM="${CONFIG_STEM#config/}"
    CONFIG_STEM="$(sanitize_path_component "$CONFIG_STEM")"
    LOG_PATH="${LOG_DIR}/${RUN_STAMP}__g${GROUP_INDEX}__${CONFIG_NAME%.*}.log"
    CONFIG_CACHE_DIR_RESOLVED="$(compile_cache_base_for_config "$CONFIG_STEM")"
    mkdir -p "$CONFIG_CACHE_DIR_RESOLVED"

    PSD_TORCH_COMPILE_CACHE_DIR="$CONFIG_CACHE_DIR_RESOLVED" \
    CONFIG_PATH="$CONFIG_PATH" \
    nohup bash -c 'torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" src/model_training.py --config "$CONFIG_PATH" --ddp true' > "$LOG_PATH" 2>&1 &
    PID="$!"
    LAST_PID="$PID"
    echo "STAGE=${STAGE}"
    echo "GROUP=${GROUP_INDEX}"
    echo "CONFIG=${CONFIG_PATH}"
    echo "LOG=${LOG_PATH}"
    echo "COMPILE_CACHE=${CONFIG_CACHE_DIR_RESOLVED}"
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
