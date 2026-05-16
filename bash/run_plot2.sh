#!/usr/bin/env bash
set -euo pipefail

# plotting_serial_wrapper_simple.sh
#
# 사용법:
#   1. 이 파일을 plotting.sh 와 같은 디렉터리에 둔다.
#
#      예:
#        psd/
#        └── bash/
#            ├── plotting.sh
#            └── plotting_serial_wrapper_simple.sh
#
#   2. 아래 CASE_* 배열을 수정한다.
#
#   3. 실행한다.
#
#        LOG_ROOT=/home/yongokhan/바탕화면/logs \
#        bash bash/plotting_serial_wrapper_simple.sh
#
# 전제:
#   plotting.sh 안의 PLOT_INPUT_SET_RAW 가 외부 입력을 받을 수 있어야 한다.
#
#   권장 수정:
#     PLOT_INPUT_SET_RAW="${PLOT_INPUT_SET_RAW:-${PLOT_INPUT_SET:?PLOT_INPUT_SET or PLOT_INPUT_SET_RAW is required}}"
#
#   직렬 실행을 원하면 plotting.sh 안의 실행부가 background/nohup 이 아니어야 한다.
#
# 배열 규칙:
#   같은 index끼리 하나의 실행 case로 묶인다.
#
#   CASE_PLOT_INPUTS[i]  : plotting 할 CSV file 또는 CSV directory tree
#   CASE_OUTPUT_ROOTS[i] : figure output root
#
# 주의:
#   실제 figure는 plotting.sh 정책에 따라 <CASE_OUTPUT_ROOTS[i]>/<case_id> 아래 저장된다.
#   PLOT_FORMAT, PLOT_OVERWRITE, PLOT_MANIFEST_NAME 등은 필요하면 외부 환경변수로 준다.

CASE_PLOT_INPUTS=(
# "/home/yongokhan/바탕화면/completed_csv/cifar10/resnet18_lif_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/cifar10/vgg11_lif_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/dvsgesture/resnet18_lif_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/deap/lif_soft/batch_0001"
  "/home/yongokhan/바탕화면/completed_csv/deap/rf_soft/batch_0001"  
#  "/home/yongokhan/바탕화면/completed_csv/dvsgesture/vgg11_lif_soft/batch_0001"
  "/home/yongokhan/바탕화면/completed_csv/s-mnist/lif_hard/batch_0001"
  "/home/yongokhan/바탕화면/completed_csv/s-mnist/lif_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/s-mnist/lif_soft_v=2/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/s-mnist/rf_hard/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/s-mnist/rf_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/s-mnist/rf_soft_v=2/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/shd/lif_hard/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/shd/lif_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/shd/lif_soft_v=2/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/shd/rf_hard/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/shd/rf_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/shd/rf_soft_v=2/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/uci-har/lif_soft/batch_0001"
# "/home/yongokhan/바탕화면/completed_csv/uci-har/rf_soft/batch_0001"
)

CASE_OUTPUT_ROOTS=(
# "/home/yongokhan/바탕화면/completed_plot/cifar10/resnet18_lif_soft"
# "/home/yongokhan/바탕화면/completed_plot/cifar10/vgg11_lif_soft"
# "/home/yongokhan/바탕화면/completed_plot/dvsgesture/resnet18_lif_soft"
# "/home/yongokhan/바탕화면/completed_plot/deap/lif_soft"
  "/home/yongokhan/바탕화면/completed_plot/deap/rf_soft" 
#  "/home/yongokhan/바탕화면/completed_plot/dvsgesture/vgg11_lif_soft" 
  "/home/yongokhan/바탕화면/completed_plot/s-mnist/lif_hard"
  "/home/yongokhan/바탕화면/completed_plot/s-mnist/lif_soft"
# "/home/yongokhan/바탕화면/completed_plot/s-mnist/lif_soft_v=2"
# "/home/yongokhan/바탕화면/completed_plot/s-mnist/rf_hard"
# "/home/yongokhan/바탕화면/completed_plot/s-mnist/rf_soft"
# "/home/yongokhan/바탕화면/completed_plot/s-mnist/rf_soft_v=2"
# "/home/yongokhan/바탕화면/completed_plot/shd/lif_hard"
# "/home/yongokhan/바탕화면/completed_plot/shd/lif_soft"
# "/home/yongokhan/바탕화면/completed_plot/shd/lif_soft_v=2"
# "/home/yongokhan/바탕화면/completed_plot/shd/rf_hard"
# "/home/yongokhan/바탕화면/completed_plot/shd/rf_soft"
# "/home/yongokhan/바탕화면/completed_plot/shd/rf_soft_v=2"
# "/home/yongokhan/바탕화면/completed_plot/uci-har/lif_soft"
# "/home/yongokhan/바탕화면/completed_plot/uci-har/rf_soft"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/plotting.sh"

if [[ ! -f "${LAUNCHER}" ]]; then
    exit 1
fi

case_count="${#CASE_PLOT_INPUTS[@]}"

if [[ "${#CASE_OUTPUT_ROOTS[@]}" -ne "${case_count}" ]]; then
    exit 1
fi

for idx in "${!CASE_PLOT_INPUTS[@]}"; do
    PLOT_INPUT_SET="${CASE_PLOT_INPUTS[$idx]}" \
    PLOT_INPUT_SET_RAW="${CASE_PLOT_INPUTS[$idx]}" \
    OUTPUT_ROOT="${CASE_OUTPUT_ROOTS[$idx]}" \
    bash "${LAUNCHER}"
done
