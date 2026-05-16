#!/usr/bin/env bash
set -euo pipefail

# 기능: dataset_psd, psd_analysis, reinterpretation 이 만든 CSV artifact 를 PNG figure 로 변환한다.
# 실행 단위: PLOT_INPUT_SET 에 적은 CSV file 또는 CSV directory tree 하나가 하나의 plotting case 이다.
# 실행 방식: 입력 case 를 현재 shell 에서 직렬 실행한다. background 실행, nohup 실행을 하지 않는다.
# 산출물: `<OUTPUT_ROOT>/<case_id>` 아래 figure file 과 plotting manifest 를 생성한다.
# 주의: 이 launcher 는 training checkpoint 를 읽지 않고, CSV artifact 만 입력으로 받는다.
# 로그: `<LOG_ROOT>/plotting/<RUN_STAMP>` 아래 case별 log 와 pid.tsv 를 생성한다.

# 인수: PROJECT_ROOT
# 기능: 이 저장소의 최상위 경로를 지정한다.
# 줄 수 있는 값: 절대경로 또는 현재 작업 위치 기준 상대경로. 기본값은 이 스크립트의 부모 디렉터리이다.
# 값의 의미: Python module 실행 시 src/ 를 import 할 기준 경로이다.
# 값 주는법: `PROJECT_ROOT=/absolute/path/to/psd bash bash/plotting.sh` 처럼 준다.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 인수: PYTHON_BIN
# 기능: plotting 을 실행할 Python executable 을 지정한다.
# 줄 수 있는 값: `python3`, `python`, conda 환경의 `/path/to/python` 같은 실행 가능 파일.
# 값의 의미: 지정한 Python 으로 `src.plotting` module 을 실행한다.
# 값 주는법: `PYTHON_BIN=/home/user/miniconda3/envs/snn/bin/python bash bash/plotting.sh` 처럼 준다.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 인수: PLOT_INPUT_SET 또는 PLOT_INPUT_SET_RAW
# 기능: figure 로 변환할 CSV file 또는 CSV directory tree 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 path 목록. 각 path 는 단일 `.csv` file 이거나 CSV file 을 포함한 directory 이다.
# 값의 의미: 각 input path 를 독립 plotting case 로 처리한다.
# 값 주는법: `PLOT_INPUT_SET="/abs/run1/csv /abs/run2/csv" bash bash/plotting.sh` 처럼 준다.
# 외부 wrapper 에서는 `PLOT_INPUT_SET` 또는 `PLOT_INPUT_SET_RAW` 중 하나를 넘기면 된다.
PLOT_INPUT_SET_RAW="${PLOT_INPUT_SET_RAW:-${PLOT_INPUT_SET:?PLOT_INPUT_SET or PLOT_INPUT_SET_RAW is required}}"

# 인수: OUTPUT_ROOT
# 기능: 생성 figure 와 manifest 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 경로 문자열. 기본값은 `${PROJECT_ROOT}/figures` 이다.
# 값의 의미: 실제 plotting 산출물 root 는 `<OUTPUT_ROOT>/<case_id>` 이다.
# 값 주는법: `OUTPUT_ROOT=/absolute/path/to/figures` 처럼 준다.
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/figures}"

# 인수: LOG_ROOT
# 기능: launcher log 와 pid.tsv 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: 실제 log directory 는 `<LOG_ROOT>/plotting/<RUN_STAMP>` 이다.
# 값 주는법: `LOG_ROOT=/absolute/path/to/logs` 처럼 준다.
LOG_ROOT="/home/yongokhan/바탕화면/logs"

# 인수: RUN_STAMP
# 기능: 한 번 실행한 launcher 묶음을 구분하는 실행 표식을 지정한다.
# 줄 수 있는 값: 파일명에 안전한 문자열. 기본값은 `YYYYmmdd_HHMMSS` 형식의 현재 시각이다.
# 값의 의미: 같은 RUN_STAMP 를 쓰면 같은 실행 묶음의 log directory 에 기록된다.
# 값 주는법: `RUN_STAMP=20260430_120000` 처럼 준다.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 인수: PLOT_FORMAT
# 기능: figure 저장 형식을 지정한다.
# 줄 수 있는 값: 현재 parser 선택지는 `png` 하나이다.
# 값의 의미: 모든 figure 를 PNG 형식으로 저장한다.
# 값 주는법: `PLOT_FORMAT=png` 처럼 준다.
PLOT_FORMAT="${PLOT_FORMAT:-png}"

# 인수: PLOT_OVERWRITE
# 기능: 이미 존재하는 figure 를 덮어쓸지 지정한다.
# 줄 수 있는 값: 참 값 `1`, `true`; 거짓 값은 그 외 값 또는 빈 값이다.
# 값의 의미: 참이면 Python 에 `--overwrite` flag 를 붙이고, 거짓이면 기존 figure 보존 정책을 따른다.
# 값 주는법: `PLOT_OVERWRITE=1` 또는 `PLOT_OVERWRITE=true` 처럼 준다.
PLOT_OVERWRITE="${PLOT_OVERWRITE:-1}"

# 인수: PLOT_MANIFEST_NAME
# 기능: plotting manifest file 이름을 지정한다.
# 줄 수 있는 값: file name 문자열. 기본값은 `plotting_manifest.csv` 이다.
# 값의 의미: rendering 결과 index 와 figure path 기록에 쓰는 manifest 이름이다.
# 값 주는법: `PLOT_MANIFEST_NAME=plotting_manifest.csv` 처럼 준다.
PLOT_MANIFEST_NAME="${PLOT_MANIFEST_NAME:-plotting_manifest.csv}"

LOG_DIR="${LOG_ROOT}/plotting/${RUN_STAMP}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

PLOT_INPUT_SET_NORMALIZED="${PLOT_INPUT_SET_RAW//,/ }"
read -r -a PLOT_INPUT_SET <<< "${PLOT_INPUT_SET_NORMALIZED}"
PID_FILE="${LOG_DIR}/pid.tsv"
printf 'run_stamp\tstage\tcase_id\tslot_id\tpid\tlog_path\tcommand\n' > "${PID_FILE}"

_sanitize_token() {
    local raw="$1"
    printf '%s' "${raw}" | tr '/:|, ' '____' | tr -cs '[:alnum:]_.=-' '_'
}

for input_path in "${PLOT_INPUT_SET[@]}"; do
    input_path="${input_path:?plot input path is required}"
    case_id="$(_sanitize_token "$(basename "$(dirname "${input_path}")")__$(basename "${input_path}")")"
    slot_id="${case_id}__plot"
    log_file="${LOG_DIR}/plotting__${RUN_STAMP}__${slot_id}.log"
    cmd=("${PYTHON_BIN}" -m src.plotting
        --input "${input_path}"
        --output_root "${OUTPUT_ROOT}/${case_id}"
        --format "${PLOT_FORMAT}"
        --manifest_name "${PLOT_MANIFEST_NAME}")
    if [[ "${PLOT_OVERWRITE}" == "1" || "${PLOT_OVERWRITE}" == "true" ]]; then
        cmd+=(--overwrite)
    fi
    printf -v command_text '%q ' "${cmd[@]}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN_STAMP}" "plotting" "${case_id}" "${slot_id}" "-" "${log_file}" "${command_text}" >> "${PID_FILE}"
    "${cmd[@]}" > "${log_file}" 2>&1
done
