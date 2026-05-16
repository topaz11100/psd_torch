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
  "/home/yongokhan/바탕화면/0505_base/shd/checkpoints/shd_lif_soft_fixed_256_128_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l1-0.001_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"
  "/home/yongokhan/바탕화면/0505_base/shd/checkpoints/shd_lif_soft_fixed_256_128_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l10_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"
  "/home/yongokhan/바탕화면/0505_base/shd/checkpoints/shd_lif_soft_fixed_256_128_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l10.001_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"
  "/home/yongokhan/바탕화면/0505_base/s-mnist/checkpoints/s-mnist_lif_soft_fixed_128_128_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l1-0.001_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"
  "/home/yongokhan/바탕화면/0505_base/s-mnist/checkpoints/s-mnist_lif_soft_fixed_128_128_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l10_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"
  "/home/yongokhan/바탕화면/0505_base/s-mnist/checkpoints/s-mnist_lif_soft_fixed_128_128_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l10.001_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"
)

CASE_DATASETS=(
  "shd"
  "shd"
  "shd"
  "s-mnist"
  "s-mnist"
  "s-mnist"
)

CASE_OUTPUT_ROOTS=(
  "/home/yongokhan/바탕화면/0505_base/shd/csv/L1_minus"
  "/home/yongokhan/바탕화면/0505_base/shd/csv/vanila"
  "/home/yongokhan/바탕화면/0505_base/shd/csv/L1_plus"
  "/home/yongokhan/바탕화면/0505_base/s-mnist/csv/L1_minus"
  "/home/yongokhan/바탕화면/0505_base/s-mnist/csv/vanila"
  "/home/yongokhan/바탕화면/0505_base/s-mnist/csv/L1_plus"
)

CASE_ANAL_BATCHES=(
  "128"
  "128"
  "128"
  "128"
  "128"
  "128"
)

CASE_GPU_SETS=(
   "0"
   "1"
   "1"
   "1"
   "0"
   "0"
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
    bash "${LAUNCHER}" &
    sleep 1
done
