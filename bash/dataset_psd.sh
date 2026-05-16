#!/usr/bin/env bash
set -euo pipefail

# 기능: prepared dataset 자체의 PSD baseline CSV 를 생성한다.
# 실행 단위: DATASET_PSD_SET 의 `<dataset>|<gpu>` 원소 하나가 하나의 child job 이다.
# 실행 방식: 모든 원소를 nohup background process 로 즉시 실행하고, parent launcher 는 종료를 기다리지 않는다.
# 분석 범위: dataset 전체 scope 와 prepared probe scope 처리는 Python program 내부 정책을 따른다.
# 산출물: `<OUTPUT_ROOT>/dataset_psd/<dataset>` 아래 category별 CSV 를 생성한다.
# 로그: `<LOG_ROOT>/dataset_psd/<RUN_STAMP>` 아래 job별 log 와 pid.tsv 를 생성한다.

# 인수: PROJECT_ROOT
# 기능: 이 저장소의 최상위 경로를 지정한다.
# 줄 수 있는 값: 절대경로 또는 현재 작업 위치 기준 상대경로. 기본값은 이 스크립트의 부모 디렉터리이다.
# 값의 의미: Python module 실행 시 src/ 를 import 할 기준 경로이다.
# 값 주는법: `PROJECT_ROOT=/absolute/path/to/psd bash bash/dataset_psd.sh` 처럼 준다.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 인수: PYTHON_BIN
# 기능: child job 을 실행할 Python executable 을 지정한다.
# 줄 수 있는 값: `python3`, `python`, conda 환경의 `/path/to/python` 같은 실행 가능 파일.
# 값의 의미: 지정한 Python 으로 `src.dataset_psd` module 을 실행한다.
# 값 주는법: `PYTHON_BIN=/home/user/miniconda3/envs/snn/bin/python bash bash/dataset_psd.sh` 처럼 준다.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 인수: PREP_ROOT
# 기능: prepared dataset bundle 을 읽을 root directory 를 지정한다.
# 줄 수 있는 값: 절대경로. 기본값은 `/home/yongokhan/바탕화면/prep_data` 이다.
# 값의 의미: 입력 prepared dataset path 는 `<PREP_ROOT>/<dataset>` 이다.
# 값 주는법: `PREP_ROOT=/absolute/path/to/prep_data` 처럼 준다.
PREP_ROOT="${PREP_ROOT:-/home/yongokhan/바탕화면/prep_data}"

# 인수: OUTPUT_ROOT
# 기능: dataset PSD CSV 산출물을 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: 실제 산출물 root 는 `<OUTPUT_ROOT>/dataset_psd/<dataset>` 이다.
# 값 주는법: `OUTPUT_ROOT=/absolute/path/to/outputs` 처럼 준다.
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

# 인수: LOG_ROOT
# 기능: launcher log 와 pid.tsv 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: 실제 log directory 는 `<LOG_ROOT>/dataset_psd/<RUN_STAMP>` 이다.
# 값 주는법: `LOG_ROOT=/absolute/path/to/logs` 처럼 준다.
LOG_ROOT="${LOG_ROOT:?LOG_ROOT is required}"

# 인수: RUN_STAMP
# 기능: 한 번 실행한 launcher 묶음을 구분하는 실행 표식을 지정한다.
# 줄 수 있는 값: 파일명에 안전한 문자열. 기본값은 `YYYYmmdd_HHMMSS` 형식의 현재 시각이다.
# 값의 의미: 같은 RUN_STAMP 를 쓰면 같은 실행 묶음의 log directory 에 기록된다.
# 값 주는법: `RUN_STAMP=20260430_120000` 처럼 준다.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 인수: DATASET_PSD_SET
# 기능: dataset PSD baseline 을 돌릴 dataset/GPU 조합 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 `<dataset_token>|<gpu_index>` 목록.
# 값의 모든 dataset 선택지: `cifar-10`, `cifar10-dvs`, `deap`, `dvs128-gesture`, `mnist`, `n-mnist`, `ps-mnist`, `s-cifar10`, `s-mnist`, `shd`, `ssc`.
# 값의 모든 gpu_index 선택지: 현재 서버의 CUDA device index 범위 안에 있는 0 이상의 정수이다. 예를 들어 GPU 4장이면 `0`, `1`, `2`, `3` 이다.
# 값의 의미: `s-mnist|0` 은 s-mnist dataset PSD 를 cuda:0 에서 실행한다는 뜻이다.
# 값 주는법: `DATASET_PSD_SET="s-mnist|0 shd|1"` 또는 `DATASET_PSD_SET="s-mnist|0,shd|1"` 처럼 준다.
DATASET_PSD_SET_RAW="${DATASET_PSD_SET:-cifar-10|0}"

# 인수: DATASET_PSD_BATCH_SIZE
# 기능: dataset PSD 계산의 mini-batch 크기를 지정한다.
# 줄 수 있는 값: 양의 정수. 기본값은 `128` 이다.
# 값의 의미: 한 번의 forward/PSD 누적에서 처리할 sample 수의 상한이다.
# 값 주는법: `DATASET_PSD_BATCH_SIZE=256` 처럼 준다.
DATASET_PSD_BATCH_SIZE="${DATASET_PSD_BATCH_SIZE:-128}"

# 인수: SEED
# 기능: probe subset 구성과 재현성 metadata 에 사용할 seed 를 지정한다.
# 줄 수 있는 값: 정수. 기본값은 `0` 이다.
# 값의 의미: dataset baseline 내부의 seed 기반 선택을 고정한다.
# 값 주는법: `SEED=0` 처럼 준다.
SEED="${SEED:-0}"

# 인수: NUM_WORKERS
# 기능: DataLoader worker 수를 지정한다.
# 줄 수 있는 값: 0 이상의 정수. 기본값은 `0` 이다.
# 값의 의미: `0` 은 main process 로 data loading 을 수행하고, 양수는 worker process 수를 뜻한다.
# 값 주는법: `NUM_WORKERS=4` 처럼 준다.
NUM_WORKERS="${NUM_WORKERS:-0}"

# PSD extractor policy: dataset PSD analysis is exact-only.
# userbin analysis and userbin CSV artifacts are intentionally disabled.

LOG_DIR="${LOG_ROOT}/dataset_psd/${RUN_STAMP}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

DATASET_PSD_SET_NORMALIZED="${DATASET_PSD_SET_RAW//,/ }"
read -r -a DATASET_PSD_SET <<< "${DATASET_PSD_SET_NORMALIZED}"
PID_FILE="${LOG_DIR}/pid.tsv"
printf 'run_stamp\tstage\tcase_id\tslot_id\tpid\tlog_path\tcommand\n' > "${PID_FILE}"

# 기능: log file name 과 pid.tsv key 에 안전하게 쓸 수 있도록 token 을 정규화한다.
_sanitize_token() {
    local raw="$1"
    printf '%s' "${raw}" | tr '/:|, ' '____' | tr -cs '[:alnum:]_.=-' '_'
}

# 기능: DATASET_PSD_SET 원소가 정확히 `<dataset>|<gpu>` 두 field 인지 검사한다.
_assert_pipe_field_count() {
    local label="$1" item="$2" expected="$3" actual
    actual="$(awk -F'|' '{print NF}' <<< "${item}")"
    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "[dataset_psd.sh] ${label} entries must contain exactly ${expected} pipe-separated fields: ${item}" >&2
        exit 1
    fi
}

launch_count=0
for item in "${DATASET_PSD_SET[@]}"; do
    _assert_pipe_field_count DATASET_PSD_SET "${item}" 2
    IFS='|' read -r dataset gpu_index <<< "${item}"
    dataset="${dataset:?dataset is required}"
    gpu_index="${gpu_index:?gpu index is required}"
    case_id="$(_sanitize_token "${dataset}")"
    slot_id="${case_id}__gpu${gpu_index}"
    log_file="${LOG_DIR}/dataset_psd__${RUN_STAMP}__${slot_id}.log"
    cmd=("${PYTHON_BIN}" -m src.dataset_psd
        # Python 인수: --dataset
        # 기능: dataset PSD baseline 대상 dataset token 을 전달한다.
        # 줄 수 있는 값: DATASET_PSD_SET 의 dataset_token field.
        # 값의 의미: `<PREP_ROOT>/<dataset>` prepared bundle 을 읽는다.
        --dataset "${dataset}"
        # Python 인수: --prep_root
        # 기능: prepared dataset root 를 전달한다.
        # 줄 수 있는 값: 절대경로.
        # 값의 의미: prepared dataset path 는 `<prep_root>/<dataset>` 이다.
        --prep_root "${PREP_ROOT}"
        # Python 인수: --output_root
        # 기능: dataset PSD CSV 출력 root 를 전달한다.
        # 줄 수 있는 값: 경로 문자열.
        # 값의 의미: category별 CSV 가 이 directory 아래에 저장된다.
        --output_root "${OUTPUT_ROOT}/dataset_psd/${dataset}"
        # Python 인수: --batch_size
        # 기능: mini-batch 크기를 전달한다.
        # 줄 수 있는 값: 양의 정수.
        # 값의 의미: 한 번에 처리할 sample 수의 상한이다.
        --batch_size "${DATASET_PSD_BATCH_SIZE}"
        # Python 인수: --gpu_index
        # 기능: CUDA device index 를 전달한다.
        # 줄 수 있는 값: 0 이상의 정수이며 현재 서버의 CUDA device 범위 안이어야 한다.
        # 값의 의미: dataset PSD baseline 계산을 수행할 GPU 이다.
        --gpu_index "${gpu_index}"
        # Python 인수: --seed
        # 기능: seed 를 전달한다.
        # 줄 수 있는 값: 정수.
        # 값의 의미: probe subset 구성과 metadata 기록에 사용된다.
        --seed "${SEED}"
        # Python 인수: --num_workers
        # 기능: DataLoader worker 수를 전달한다.
        # 줄 수 있는 값: 0 이상의 정수.
        # 값의 의미: dataset loading 에 사용할 worker process 수이다.
        --num_workers "${NUM_WORKERS}")
    printf -v command_text '%q ' "${cmd[@]}"
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN_STAMP}" "dataset_psd" "${case_id}" "${slot_id}" "$!" "${log_file}" "${command_text}" >> "${PID_FILE}"
    launch_count=$((launch_count + 1))
done

printf '[dataset_psd.sh] launched %d job(s); pid file: %s\n' "${launch_count}" "${PID_FILE}"
