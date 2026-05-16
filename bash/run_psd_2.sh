#!/usr/bin/env bash
set -euo pipefail

# psd_analysis_serial_wrapper_simple.sh
#
# 사용법:
#   1. 이 파일을 psd_analysis.sh 와 같은 디렉터리에 둔다.
#
#      예:
#        psd/
#        └── bash/
#            ├── psd_analysis.sh
#            └── psd_analysis_serial_wrapper_simple.sh
#
#   2. 아래 CASE_* 배열을 수정한다.
#
#   3. 실행한다.
#
#        bash bash/psd_analysis_serial_wrapper_simple.sh
#
#   4. 필요한 공통 환경변수는 실행 전에 외부에서 준다.
#
#        LOG_ROOT=/home/yongokhan/바탕화면/logs \
#        PREP_ROOT=/home/yongokhan/바탕화면/prep_data \
#        bash bash/psd_analysis_serial_wrapper_simple.sh
#
# 전제:
#   psd_analysis.sh 안의 CHECKPOINT_SET_RAW 가 외부 입력을 받을 수 있어야 한다.
#
#   권장 수정:
#     CHECKPOINT_SET_RAW="${CHECKPOINT_SET_RAW:-${CHECKPOINT_SET:?CHECKPOINT_SET or CHECKPOINT_SET_RAW is required}}"
#
# 배열 규칙:
#   같은 index끼리 하나의 실행 case로 묶인다.
#
#   CASE_CHECKPOINTS[i]  : checkpoint .pt 파일 또는 .pt-only checkpoint directory
#   CASE_DATASETS[i]     : dataset token
#   CASE_OUTPUT_ROOTS[i] : analysis output root
#   CASE_ANAL_BATCHES[i] : analysis batch size
#   CASE_GPU_SETS[i]     : 사용할 GPU index 또는 GPU index set
#
# 주의:
#   이 wrapper는 case 단위 직렬 실행만 담당한다.
#   실제 psd_analysis 실행 옵션의 세부 처리는 같은 폴더의 psd_analysis.sh 가 담당한다.

CASE_CHECKPOINTS=(
  "/home/yongokhan/바탕화면/0504_image/cifar-100/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/s-mnist/rf_soft/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/s-mnist/rf_soft_v=2/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/shd/lif_hard/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/shd/lif_soft/checkpoint"
# "/home/yongokhan/바탕화면/completed_pt/shd/lif_soft_v=2/checkpoint"
# "/home/yongokhan/바탕화면/completed_pt/shd/rf_hard/checkpoint"
# "/home/yongokhan/바탕화면/completed_pt/shd/rf_soft/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/shd/rf_soft_v=2/checkpoint"
# "/home/yongokhan/바탕화면/completed_pt/uci-har/lif_soft/checkpoint"
# "/home/yongokhan/바탕화면/completed_pt/uci-har/rf_soft/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/deap/lif_soft/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/deap/rf_soft/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/s-mnist/spikegru/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/shd/spikegru/checkpoint"
#  "/home/yongokhan/바탕화면/completed_pt/uci-har/spikegru/checkpoints"
)

CASE_DATASETS=(
  "cifar-100"
# "s-mnist"
# "s-mnist"
# "shd"
# "shd"
# "shd"
# "shd"
# "shd"
#  "shd"
# "uci-har"
# "uci-har"
#  "s-mnist"
#  "shd"
)

CASE_OUTPUT_ROOTS=(
  "/home/yongokhan/바탕화면/completed_csv/cifar-100/resnet18_lif_soft"
#  "/home/yongokhan/바탕화면/completed_csv/s-mnist/rf_soft"
#  "/home/yongokhan/바탕화면/completed_csv/s-mnist/rf_soft_v=2"
#  "/home/yongokhan/바탕화면/completed_csv/shd/lif_hard"
#  "/home/yongokhan/바탕화면/completed_csv/shd/lif_soft"
# "/home/yongokhan/바탕화면/completed_csv/shd/lif_soft_v=2"
# "/home/yongokhan/바탕화면/completed_csv/shd/rf_hard"
# "/home/yongokhan/바탕화면/completed_csv/shd/rf_soft"
#  "/home/yongokhan/바탕화면/completed_csv/shd/rf_soft_v=2"
# "/home/yongokhan/바탕화면/completed_csv/uci-har/lif_soft"
# "/home/yongokhan/바탕화면/completed_csv/uci-har/rf_soft"
#  "/home/yongokhan/바탕화면/completed_csv/s-mnist/spikegru"
#  "/home/yongokhan/바탕화면/completed_csv/shd/spikegru"
# "/home/yongokhan/바탕화면/completed_csv/uci-har/spikegru"
)

CASE_ANAL_BATCHES=(
#  "512"
#  "512"
#  "512"
#  "512"
#  "512"
#  "512"
#  "512"
#  "512"
# "512"
# "512"
#  "512"
  "128"
)

CASE_GPU_SETS=(
# "1"
# "1"
# "1"
# "1"
# "1"
# "1"
# "1"
#"1"
#  "1"
  "1"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/psd_analysis.sh"

if [[ ! -f "${LAUNCHER}" ]]; then
    exit 1
fi

case_count="${#CASE_CHECKPOINTS[@]}"

if [[ "${#CASE_DATASETS[@]}" -ne "${case_count}" ]]; then
    exit 1
fi

if [[ "${#CASE_OUTPUT_ROOTS[@]}" -ne "${case_count}" ]]; then
    exit 1
fi

if [[ "${#CASE_ANAL_BATCHES[@]}" -ne "${case_count}" ]]; then
    exit 1
fi

if [[ "${#CASE_GPU_SETS[@]}" -ne "${case_count}" ]]; then
    exit 1
fi

for idx in "${!CASE_CHECKPOINTS[@]}"; do
    CHECKPOINT_SET="${CASE_CHECKPOINTS[$idx]}" \
    CHECKPOINT_SET_RAW="${CASE_CHECKPOINTS[$idx]}" \
    DATASET="${CASE_DATASETS[$idx]}" \
    OUTPUT_ROOT="${CASE_OUTPUT_ROOTS[$idx]}" \
    ANAL_BATCH="${CASE_ANAL_BATCHES[$idx]}" \
    GPU_INDEX_SET="${CASE_GPU_SETS[$idx]}" \
    bash "${LAUNCHER}"
done
