#!/usr/bin/env bash
set -euo pipefail

# 기능: 기존 논문/저자 코드 계열 실험을 project CSV schema 에 맞게 재해석하는 driver 를 실행한다.
# 실행 단위: 이 launcher 는 한 번 실행할 때 reinterpretation driver child job 하나를 실행한다.
# 실행 방식: child job 을 nohup background process 로 실행하고, parent launcher 는 종료를 기다리지 않는다.
# 산출물: `<OUTPUT_ROOT>/reinterpretation/<experiment_id>` 아래 metadata 와 CSV placeholder 또는 hook 산출물을 생성한다.
# 주의: 이 launcher 는 main model_training, psd_analysis, plotting stage 를 대신하지 않는다.
# 로그: `<LOG_ROOT>/reinterpretation/<RUN_STAMP>` 아래 log 와 pid.tsv 를 생성한다.

# 인수: PROJECT_ROOT
# 기능: 이 저장소의 최상위 경로를 지정한다.
# 줄 수 있는 값: 절대경로 또는 현재 작업 위치 기준 상대경로. 기본값은 이 스크립트의 부모 디렉터리이다.
# 값의 의미: Python module 실행 시 src/ 를 import 할 기준 경로이다.
# 값 주는법: `PROJECT_ROOT=/absolute/path/to/psd bash bash/reinterpretation.sh` 처럼 준다.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 인수: PYTHON_BIN
# 기능: child job 을 실행할 Python executable 을 지정한다.
# 줄 수 있는 값: `python3`, `python`, conda 환경의 `/path/to/python` 같은 실행 가능 파일.
# 값의 의미: 지정한 Python 으로 `src.reinterpretation.driver` module 을 실행한다.
# 값 주는법: `PYTHON_BIN=/home/user/miniconda3/envs/snn/bin/python bash bash/reinterpretation.sh` 처럼 준다.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 인수: OUTPUT_ROOT
# 기능: reinterpretation 산출물을 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 절대경로 또는 경로 문자열.
# 값의 의미: 실제 산출물 root 는 `<OUTPUT_ROOT>/reinterpretation/<experiment_id>` 이다.
# 값 주는법: `OUTPUT_ROOT=/absolute/path/to/outputs` 처럼 준다.
OUTPUT_ROOT="/home/yongokhan/바탕화면/re"

# 인수: LOG_ROOT
# 기능: launcher log 와 pid.tsv 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: 실제 log directory 는 `<LOG_ROOT>/reinterpretation/<RUN_STAMP>` 이다.
# 값 주는법: `LOG_ROOT=/absolute/path/to/logs` 처럼 준다.
LOG_ROOT="/home/yongokhan/바탕화면/logs"

# 인수: RUN_STAMP
# 기능: 한 번 실행한 launcher 묶음을 구분하는 실행 표식을 지정한다.
# 줄 수 있는 값: 파일명에 안전한 문자열. 기본값은 `YYYYmmdd_HHMMSS` 형식의 현재 시각이다.
# 값의 의미: 같은 RUN_STAMP 를 쓰면 같은 실행 묶음의 log directory 에 기록된다.
# 값 주는법: `RUN_STAMP=20260430_120000` 처럼 준다.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 인수: RUN_NEED_HIGH
# 기능: Need-High 계열 reinterpretation case 실행 여부를 지정한다.
# 줄 수 있는 값: 참 값 `1`, `true`, `t`, `yes`, `y`, `on`; 거짓 값 `0`, `false`, `f`, `no`, `n`, `off`.
# 값의 의미: 참이면 experiment id `need_high` 를 실행 대상으로 포함한다.
# 값 주는법: `RUN_NEED_HIGH=true` 처럼 준다.
RUN_NEED_HIGH="${RUN_NEED_HIGH:-true}"

# 인수: RUN_DRF
# 기능: D-RF 계열 reinterpretation case 실행 여부를 지정한다.
# 줄 수 있는 값: 참 값 `1`, `true`, `t`, `yes`, `y`, `on`; 거짓 값 `0`, `false`, `f`, `no`, `n`, `off`.
# 값의 의미: 참이면 experiment id `drf` 를 실행 대상으로 포함한다.
# 값 주는법: `RUN_DRF=true` 처럼 준다.
RUN_DRF="${RUN_DRF:-false}"

# 인수: RUN_DH_SNN
# 기능: DH-SNN 계열 reinterpretation case 실행 여부를 지정한다.
# 줄 수 있는 값: 참 값 `1`, `true`, `t`, `yes`, `y`, `on`; 거짓 값 `0`, `false`, `f`, `no`, `n`, `off`.
# 값의 의미: 참이면 experiment id `dh_snn` 를 실행 대상으로 포함한다.
# 값 주는법: `RUN_DH_SNN=true` 처럼 준다.
RUN_DH_SNN="${RUN_DH_SNN:-false}"

# 인수: GPU_MAP
# 기능: reinterpretation experiment id 별 GPU index 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 콤마 구분 `experiment_id:gpu_index` 목록.
# 값의 모든 experiment_id 선택지: `need_high`, `drf`, `dh_snn`.
# 값의 모든 gpu_index 선택지: 0 이상의 정수이다.
# 값의 의미: `need_high:0,drf:1` 은 need_high 를 cuda:0, drf 를 cuda:1 에 배정한다는 뜻이다.
# 값 주는법: `GPU_MAP="need_high:0,drf:1,dh_snn:0"` 처럼 준다. 실행 flag 가 참인 experiment_id 는 반드시 GPU_MAP 에 있어야 한다.
GPU_MAP="need_high:0,drf:1,dh_snn:0"

# 인수: SEED_BUNDLE
# 기능: reinterpretation bundle 에 기록할 seed bundle identifier 를 지정한다.
# 줄 수 있는 값: 문자열 또는 정수 문자열. 기본값은 `0` 이다.
# 값의 의미: run_id 와 metadata 에 기록되는 seed bundle 표식이다.
# 값 주는법: `SEED_BUNDLE=0` 또는 `SEED_BUNDLE=paper_seed_a` 처럼 준다.
SEED_BUNDLE="${SEED_BUNDLE:-0}"

# 인수: USERBIN_EDGES
# 기능: userbin PSD 의 normalized-frequency bin 경계를 지정한다.
# 줄 수 있는 값: 공백으로 구분한 증가하는 실수열. 첫 값은 `0.0`, 마지막 값은 `0.5` 이어야 하며 최소 두 값이 필요하다.
# 값의 모든 기본 경계: `0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50`.
# 값의 의미: reinterpretation metadata 와 hook CSV 의 userbin frequency aggregation 경계로 사용된다.
# 값 주는법: `USERBIN_EDGES="0.0 0.1 0.2 0.3 0.4 0.5"` 처럼 준다. 비우면 Python 기본 경계를 사용한다.
USERBIN_EDGES="0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50"

LOG_DIR="${LOG_ROOT}/reinterpretation/${RUN_STAMP}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
PID_FILE="${LOG_DIR}/pid.tsv"
printf 'run_stamp\tstage\tcase_id\tslot_id\tpid\tlog_path\tcommand\n' > "${PID_FILE}"

log_file="${LOG_DIR}/reinterpretation__${RUN_STAMP}.log"
cmd=("${PYTHON_BIN}" -m src.reinterpretation.driver
    # Python 인수: --run_need_high
    # 기능: need_high case 실행 여부를 전달한다.
    # 줄 수 있는 값: RUN_NEED_HIGH 설명의 참/거짓 token.
    # 값의 의미: 참이면 need_high reinterpretation output 을 생성한다.
    --run_need_high "${RUN_NEED_HIGH}"
    # Python 인수: --run_drf
    # 기능: drf case 실행 여부를 전달한다.
    # 줄 수 있는 값: RUN_DRF 설명의 참/거짓 token.
    # 값의 의미: 참이면 drf reinterpretation output 을 생성한다.
    --run_drf "${RUN_DRF}"
    # Python 인수: --run_dh_snn
    # 기능: dh_snn case 실행 여부를 전달한다.
    # 줄 수 있는 값: RUN_DH_SNN 설명의 참/거짓 token.
    # 값의 의미: 참이면 dh_snn reinterpretation output 을 생성한다.
    --run_dh_snn "${RUN_DH_SNN}"
    # Python 인수: --gpu_map
    # 기능: experiment id 별 GPU index mapping 을 전달한다.
    # 줄 수 있는 값: GPU_MAP 설명의 `experiment_id:gpu_index` 콤마 목록.
    # 값의 의미: 각 enabled reinterpretation case 의 device 배정표이다.
    --gpu_map "${GPU_MAP}"
    # Python 인수: --output_root
    # 기능: 산출물 root 를 전달한다.
    # 줄 수 있는 값: 경로 문자열.
    # 값의 의미: reinterpretation metadata 와 CSV 의 저장 root 이다.
    --output_root "${OUTPUT_ROOT}"
    # Python 인수: --log_root
    # 기능: driver 내부 log root 를 전달한다.
    # 줄 수 있는 값: 경로 문자열.
    # 값의 의미: 이 launcher 의 log directory 를 driver 에도 기록 root 로 알려준다.
    --log_root "${LOG_DIR}"
    # Python 인수: --seed_bundle
    # 기능: seed bundle identifier 를 전달한다.
    # 줄 수 있는 값: 문자열 또는 정수 문자열.
    # 값의 의미: run_id 와 metadata 에 기록된다.
    --seed_bundle "${SEED_BUNDLE}")
if [[ -n "${USERBIN_EDGES}" ]]; then
    read -r -a userbin_args <<< "${USERBIN_EDGES}"
    # Python 인수: --userbin_edges
    # 기능: userbin PSD 경계를 전달한다.
    # 줄 수 있는 값: USERBIN_EDGES 설명의 증가하는 normalized-frequency 실수열.
    # 값의 의미: exact PSD 계산 뒤 frequency aggregation 구간으로 사용된다.
    cmd+=(--userbin_edges "${userbin_args[@]}")
fi
printf -v command_text '%q ' "${cmd[@]}"
nohup "${cmd[@]}" > "${log_file}" 2>&1 &
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN_STAMP}" "reinterpretation" "reinterpretation" "reinterpretation" "$!" "${log_file}" "${command_text}" >> "${PID_FILE}"

printf '[reinterpretation.sh] launched 1 job; pid file: %s\n' "${PID_FILE}"
