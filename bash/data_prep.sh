 #!/usr/bin/env bash
set -euo pipefail

# 기능: raw dataset 을 prepared dataset bundle 로 변환한다.
# 실행 단위: DATA_PREP_SET 에 적은 dataset token 하나가 하나의 child job 이다.
# 실행 방식: 각 child job 을 nohup background process 로 실행하고, parent launcher 는 종료를 기다리지 않는다.
# 산출물: <PREP_ROOT>/<dataset>/ 아래 prepared bundle, manifest, axis metadata 를 생성한다.
# 로그: <LOG_ROOT>/data_prep/<RUN_STAMP>/ 아래 job별 log 와 pid.tsv 를 생성한다.

# 인수: PROJECT_ROOT
# 기능: 이 저장소의 최상위 경로를 지정한다. src/ 와 bash/ 를 포함하는 디렉터리여야 한다.
# 줄 수 있는 값: 절대경로 또는 현재 작업 위치 기준 상대경로. 기본값은 이 스크립트의 부모 디렉터리이다.
# 값의 의미: Python module 실행 시 `-m src.data_prep` 가 import 되는 기준 프로젝트 경로이다.
# 값 주는법: `PROJECT_ROOT=/absolute/path/to/psd bash bash/data_prep.sh` 처럼 환경변수로 준다.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 인수: PYTHON_BIN
# 기능: child job 을 실행할 Python executable 을 지정한다.
# 줄 수 있는 값: `python3`, `python`, conda 환경의 `/path/to/python` 같은 실행 가능 파일.
# 값의 의미: 지정한 Python 으로 `src.data_prep` module 을 실행한다.
# 값 주는법: `PYTHON_BIN=/home/user/miniconda3/envs/snn/bin/python bash bash/data_prep.sh` 처럼 준다.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 인수: RAW_DATA_ROOT
# 기능: 원본 dataset 파일을 찾을 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 절대경로. dataset별 원본 파일이 이 경로 아래에 있어야 한다.
# 값의 의미: `src.data_prep --raw_data_root` 로 그대로 전달된다.
# 값 주는법: `RAW_DATA_ROOT=/absolute/path/to/raw_data` 처럼 준다.
RAW_DATA_ROOT="/home/yongokhan/바탕화면/data"

# 인수: PREP_ROOT
# 기능: prepared dataset bundle 을 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 절대경로. 기본값은 `/home/yongokhan/바탕화면/prep_data` 이다.
# 값의 의미: 결과는 `<PREP_ROOT>/<dataset>` 구조로 저장된다.
# 값 주는법: `PREP_ROOT=/absolute/path/to/prep_data` 처럼 준다.
PREP_ROOT="/home/yongokhan/바탕화면/prep_data"

# 인수: LOG_ROOT
# 기능: launcher log 와 pid.tsv 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 절대경로 또는 경로 문자열.
# 값의 의미: 실제 log directory 는 `<LOG_ROOT>/data_prep/<RUN_STAMP>` 이다.
# 값 주는법: `LOG_ROOT=/absolute/path/to/logs` 처럼 준다.
LOG_ROOT="/home/yongokhan/바탕화면/logs"

# 인수: RUN_STAMP
# 기능: 한 번 실행한 launcher 묶음을 구분하는 실행 표식을 지정한다.
# 줄 수 있는 값: 파일명에 안전한 문자열. 기본값은 `YYYYmmdd_HHMMSS` 형식의 현재 시각이다.
# 값의 의미: 같은 RUN_STAMP 를 쓰면 같은 실행 묶음의 log directory 에 기록된다.
# 값 주는법: `RUN_STAMP=20260430_120000` 처럼 준다.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 인수: DATA_PREP_SET
# 기능: 준비할 dataset token 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 dataset token 목록.
# 값의 모든 선택지: canonical token 은 `cifar-10`, `cifar10-dvs`, `deap`, `dvs128-gesture`, `mnist`, `n-mnist`, `ps-mnist`, `s-cifar10`, `s-mnist`, `shd`, `ssc` 이다.
# 값의 의미: 각 token 마다 독립 data preparation job 하나를 실행한다.
# 값 주는법: `DATA_PREP_SET="s-mnist shd"` 또는 `DATA_PREP_SET="s-mnist,shd"` 처럼 준다.
DATA_PREP_SET_RAW="s-mnist,ps-mnist,s-cifar10,shd,ssc,deap,uci-har,mnist,cifar-10,cifar-100,n-mnist,cifar10-dvs,dvs128-gesture"

# 인수: SEED
# 기능: preprocessing metadata 에 기록할 seed 를 지정한다.
# 줄 수 있는 값: 정수. 기본값은 `0` 이다.
# 값의 의미: probe/index 구성 등 seed 를 기록해야 하는 준비 단계에서 기준값으로 사용된다.
# 값 주는법: `SEED=0` 처럼 준다.
SEED="${SEED:-0}"

# 인수: FORCE_OVERWRITE
# 기능: 이미 존재하는 prepared bundle 을 덮어쓸지 지정한다.
# 줄 수 있는 값: 참 값 `1`, `true`, `t`, `yes`, `y`, `on`; 거짓 값 `0`, `false`, `f`, `no`, `n`, `off`.
# 값의 의미: 참이면 기존 `<PREP_ROOT>/<dataset>` 산출물을 교체하고, 거짓이면 기존 산출물이 있을 때 실패한다.
# 값 주는법: `FORCE_OVERWRITE=true` 처럼 준다.
FORCE_OVERWRITE="${FORCE_OVERWRITE:-true}"

LOG_DIR="${LOG_ROOT}/data_prep/${RUN_STAMP}"
mkdir -p "${PREP_ROOT}" "${LOG_DIR}"

DATA_PREP_SET_NORMALIZED="${DATA_PREP_SET_RAW//,/ }"
read -r -a DATA_PREP_SET <<< "${DATA_PREP_SET_NORMALIZED}"
PID_FILE="${LOG_DIR}/pid.tsv"
printf 'run_stamp\tstage\tcase_id\tslot_id\tpid\tlog_path\tcommand\n' > "${PID_FILE}"

# 기능: log file name 과 pid.tsv key 에 안전하게 쓸 수 있도록 dataset token 을 정규화한다.
_sanitize_token() {
    local raw="$1"
    printf '%s' "${raw}" | tr '/:|, ' '____' | tr -cs '[:alnum:]_.=-' '_'
}

launch_count=0
for dataset in "${DATA_PREP_SET[@]}"; do
    dataset="${dataset:?dataset is required}"
    case_id="$(_sanitize_token "${dataset}")"
    slot_id="${case_id}__prep"
    log_file="${LOG_DIR}/data_prep__${RUN_STAMP}__${slot_id}.log"
    cmd=("${PYTHON_BIN}" -m src.data_prep
        # Python 인수: --dataset
        # 기능: prepared bundle 을 만들 dataset token 을 지정한다.
        # 줄 수 있는 값: DATA_PREP_SET 설명의 canonical token 또는 코드가 허용하는 alias.
        # 값의 의미: `<PREP_ROOT>/<dataset>` 의 dataset 부분이 된다.
        --dataset "${dataset}"
        # Python 인수: --raw_data_root
        # 기능: 원본 dataset root 를 전달한다.
        # 줄 수 있는 값: 절대경로.
        # 값의 의미: 원본 파일을 찾는 기준 root 이다.
        --raw_data_root "${RAW_DATA_ROOT}"
        # Python 인수: --prep_root
        # 기능: prepared bundle 저장 root 를 전달한다.
        # 줄 수 있는 값: 절대경로.
        # 값의 의미: prepared dataset path 는 `<prep_root>/<dataset>` 이다.
        --prep_root "${PREP_ROOT}"
        # Python 인수: --seed
        # 기능: preprocessing seed 를 전달한다.
        # 줄 수 있는 값: 정수.
        # 값의 의미: manifest metadata 에 기록되는 seed 이다.
        --seed "${SEED}"
        # Python 인수: --force_overwrite
        # 기능: 기존 prepared bundle 교체 여부를 전달한다.
        # 줄 수 있는 값: FORCE_OVERWRITE 설명의 참/거짓 token.
        # 값의 의미: 참이면 기존 산출물 교체를 허용한다.
        --force_overwrite "${FORCE_OVERWRITE}")
    printf -v command_text '%q ' "${cmd[@]}"
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN_STAMP}" "data_prep" "${case_id}" "${slot_id}" "$!" "${log_file}" "${command_text}" >> "${PID_FILE}"
    launch_count=$((launch_count + 1))
done

printf '[data_prep.sh] launched %d job(s); pid file: %s\n' "${launch_count}" "${PID_FILE}"
