#!/usr/bin/env bash
set -euo pipefail

# 기능: training 이 저장한 .pt checkpoint 를 불러와 train/test full split 정확도 CSV 를 생성한다.
# 실행 단위: CHECKPOINT_SET 에서 발견한 .pt file 을 CHECKPOINTS_PER_JOB 규칙으로 묶은 batch 하나가 하나의 child job 이다.
# 실행 방식: 모든 accuracy analysis job 을 nohup background process 로 즉시 실행하고, parent launcher 는 종료를 기다리지 않는다.
# 분석 범위: checkpoint 에 저장된 model/readout 을 복원하고 prepared train/test split 전체를 추론 평가한다.
# 산출물: `<OUTPUT_ROOT>/<case_id>/checkpoint_accuracy.csv` 와 `<OUTPUT_ROOT>/<case_id>/analysis_manifest.csv` 를 생성한다.
# 주의: 이 launcher 는 training 과 figure rendering 을 실행하지 않는다.
# 로그: `<LOG_ROOT>/checkpoint_accuracy_analysis/<RUN_STAMP>` 아래 job별 log, temporary checkpoint batch directory, pid.tsv 를 생성한다.

# 인수: PROJECT_ROOT
# 기능: 이 저장소의 최상위 경로를 지정한다.
# 줄 수 있는 값: 절대경로 또는 현재 작업 위치 기준 상대경로. 기본값은 이 스크립트의 부모 디렉터리이다.
# 값의 의미: Python module 실행 시 src/ 를 import 할 기준 경로이다.
# 값 주는법: `PROJECT_ROOT=/absolute/path/to/psd bash bash/checkpoint_accuracy_analysis.sh` 처럼 준다.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 인수: PYTHON_BIN
# 기능: child job 을 실행할 Python executable 을 지정한다.
# 줄 수 있는 값: `python3`, `python`, conda 환경의 `/path/to/python` 같은 실행 가능 파일.
# 값의 의미: 지정한 Python 으로 `src.checkpoint_accuracy_analysis` module 을 실행한다.
# 값 주는법: `PYTHON_BIN=/home/user/miniconda3/envs/snn/bin/python bash bash/checkpoint_accuracy_analysis.sh` 처럼 준다.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 인수: CHECKPOINT_SET
# 기능: 평가할 checkpoint file 또는 strict .pt-only checkpoint directory 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 path 목록. 각 path 는 `.pt` file 또는 바로 아래에 `.pt` file 만 가진 directory 여야 한다.
# 값의 의미: file 이면 그 checkpoint 하나를 평가 후보로 넣고, directory 이면 바로 아래 `.pt` file 전체를 평가 후보로 넣는다.
# 값 주는법: `CHECKPOINT_SET="/abs/checkpoints/a.pt /abs/checkpoints/run_b"` 처럼 준다.
CHECKPOINT_SET_RAW="/home/yongokhan/바탕화면/new/s-mnist/checkpoints/s-mnist_lif_soft_fixed_128_64_64_temporal_membrane_e50_b256_lr0.005_seed0_reg_l10_l20_sigy_spike_spaceexact_scaledb_centerraw_redmean"

# 인수: CHECKPOINTS_PER_JOB
# 기능: 한 Python accuracy analysis process 가 받을 checkpoint 개수를 지정한다.
# 줄 수 있는 값: 양의 정수 또는 `all`. 기본값은 `all` 이다.
# 값의 모든 의미: `1` 은 checkpoint 하나당 process 하나, `N` 은 최대 N개를 temporary .pt-only directory 로 묶은 process 하나, `all` 은 발견된 전체 checkpoint 를 process 하나로 묶는다는 뜻이다.
# 값 주는법: `CHECKPOINTS_PER_JOB=1`, `CHECKPOINTS_PER_JOB=4`, `CHECKPOINTS_PER_JOB=all` 처럼 준다.
CHECKPOINTS_PER_JOB="all"

# 인수: DATASET
# 기능: checkpoint training 에 사용한 dataset token 을 지정한다.
# 줄 수 있는 값: `cifar-10`, `cifar10-dvs`, `deap`, `dvs128-gesture`, `mnist`, `n-mnist`, `ps-mnist`, `s-cifar10`, `s-mnist`, `shd`, `ssc`.
# 값의 의미: prepared train/test split 을 `<PREP_ROOT>/<DATASET>` 에서 읽고, checkpoint metadata 의 dataset 과 충돌하면 Python process 가 실패한다.
# 값 주는법: `DATASET=s-mnist` 처럼 준다.
DATASET="s-mnist"

# 인수: PREP_ROOT
# 기능: prepared dataset bundle 을 읽을 root directory 를 지정한다.
# 줄 수 있는 값: 절대경로. 기본값은 `/home/yongokhan/바탕화면/prep_data` 이다.
# 값의 의미: 입력 prepared dataset path 는 `<PREP_ROOT>/<DATASET>` 이다.
# 값 주는법: `PREP_ROOT=/absolute/path/to/prep_data` 처럼 준다.
PREP_ROOT="/home/yongokhan/바탕화면/prep_data"

# 인수: OUTPUT_ROOT
# 기능: accuracy CSV 산출물을 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: 실제 산출물 root 는 `<OUTPUT_ROOT>/<case_id>` 이다.
# 값 주는법: `OUTPUT_ROOT=/absolute/path/to/accuracy_outputs` 처럼 준다.
OUTPUT_ROOT="/home/yongokhan/바탕화면/new/s-mnist/Acc_checkpoint"

# 인수: LOG_ROOT
# 기능: launcher log 와 pid.tsv 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 경로 문자열. 기본값은 `/home/yongokhan/바탕화면/logs` 이다.
# 값의 의미: 실제 log directory 는 `<LOG_ROOT>/checkpoint_accuracy_analysis/<RUN_STAMP>` 이다.
# 값 주는법: `LOG_ROOT=/absolute/path/to/logs` 처럼 준다.
LOG_ROOT="${LOG_ROOT:-/home/yongokhan/바탕화면/logs}"

# 인수: RUN_STAMP
# 기능: 한 번 실행한 launcher 묶음을 구분하는 실행 표식을 지정한다.
# 줄 수 있는 값: 파일명에 안전한 문자열. 기본값은 `YYYYmmdd_HHMMSS` 형식의 현재 시각이다.
# 값의 의미: 같은 RUN_STAMP 를 쓰면 같은 실행 묶음의 log directory 에 기록된다.
# 값 주는법: `RUN_STAMP=20260430_120000` 처럼 준다.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 인수: ANAL_BATCH
# 기능: accuracy evaluation forward pass 당 sample 수 상한을 지정한다.
# 줄 수 있는 값: 양의 정수. 기본값은 `128` 이다.
# 값의 의미: train/test split 을 몇 sample 단위로 나누어 model forward 할지 정한다.
# 값 주는법: `ANAL_BATCH=256` 처럼 준다.
ANAL_BATCH="${ANAL_BATCH:-4096}"

# 인수: GPU_INDEX_SET
# 기능: accuracy analysis job 을 배정할 CUDA device 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 0 이상의 정수 목록. 각 값은 현재 서버의 CUDA device 범위 안이어야 한다.
# 값의 의미: job batch 순서대로 GPU 목록을 순환 배정한다. 예를 들어 `0 1` 이면 1번 job은 cuda:0, 2번 job은 cuda:1, 3번 job은 cuda:0 이다.
# 값 주는법: `GPU_INDEX_SET="0 1"` 또는 `GPU_INDEX_SET="0,1"` 처럼 준다.
GPU_INDEX_SET_RAW="${GPU_INDEX_SET:-0}"

# 인수: SEED
# 기능: accuracy analysis seed 를 선택적으로 지정한다.
# 줄 수 있는 값: 빈 값 또는 정수. 기본값은 빈 값이다.
# 값의 의미: 빈 값이면 checkpoint seed 를 쓰고, 정수를 주면 `--seed` 로 전달한다.
# 값 주는법: `SEED=0` 처럼 준다. checkpoint seed 를 쓰려면 `SEED=` 로 비워 둔다.
SEED="${SEED:-}"

# 인수: NUM_WORKERS
# 기능: DataLoader worker 수를 지정한다.
# 줄 수 있는 값: 0 이상의 정수. 기본값은 `8` 이다.
# 값의 의미: `0` 은 main process 로 data loading 을 수행하고, 양수는 worker process 수를 뜻한다.
# 값 주는법: `NUM_WORKERS=4` 처럼 준다.
NUM_WORKERS="${NUM_WORKERS:-8}"

# 인수: SPLITS
# 기능: 평가할 split 목록을 지정한다.
# 줄 수 있는 값: `train`, `test`, 또는 공백/콤마로 구분한 `train test`. 기본값은 `train test` 이다.
# 값의 의미: 지정한 split 에 대해서만 정확도를 계산한다.
# 값 주는법: `SPLITS="train test"` 또는 `SPLITS=test` 처럼 준다.
SPLITS_RAW="${SPLITS:-train test}"

LOG_DIR="${LOG_ROOT}/checkpoint_accuracy_analysis/${RUN_STAMP}"
BATCH_DIR_ROOT="${LOG_DIR}/checkpoint_batches"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${BATCH_DIR_ROOT}"

CHECKPOINT_SET_NORMALIZED="${CHECKPOINT_SET_RAW//,/ }"
GPU_INDEX_SET_NORMALIZED="${GPU_INDEX_SET_RAW//,/ }"
SPLITS_NORMALIZED="${SPLITS_RAW//,/ }"
read -r -a CHECKPOINT_SET <<< "${CHECKPOINT_SET_NORMALIZED}"
read -r -a GPU_INDEX_SET <<< "${GPU_INDEX_SET_NORMALIZED}"
read -r -a SPLITS <<< "${SPLITS_NORMALIZED}"
if [[ ${#GPU_INDEX_SET[@]} -lt 1 || -z "${GPU_INDEX_SET[0]:-}" ]]; then
    echo '[checkpoint_accuracy_analysis.sh] GPU_INDEX_SET must contain at least one GPU index.' >&2
    exit 1
fi
if [[ ${#SPLITS[@]} -lt 1 || -z "${SPLITS[0]:-}" ]]; then
    echo '[checkpoint_accuracy_analysis.sh] SPLITS must contain train and/or test.' >&2
    exit 1
fi

PID_FILE="${LOG_DIR}/pid.tsv"
printf 'run_stamp\tstage\tcase_id\tslot_id\tpid\tlog_path\tcommand\n' > "${PID_FILE}"

# 기능: log file name 과 pid.tsv key 에 안전하게 쓸 수 있도록 token 을 정규화한다.
_sanitize_token() {
    local raw="$1"
    printf '%s' "${raw}" | tr '/:|, ' '____' | tr -cs '[:alnum:]_.=-' '_'
}

# 기능: checkpoint 입력이 file 인지 directory 인지 검사하고, accuracy analysis 후보 .pt file absolute path 목록으로 펼친다.
_collect_checkpoint_files() {
    local entry="$1"
    if [[ -f "${entry}" ]]; then
        if [[ "${entry}" != *.pt ]]; then
            echo "[checkpoint_accuracy_analysis.sh] checkpoint file must end with .pt: ${entry}" >&2
            exit 1
        fi
        printf '%s\n' "$(realpath "${entry}")"
        return
    fi
    if [[ ! -d "${entry}" ]]; then
        echo "[checkpoint_accuracy_analysis.sh] checkpoint input does not exist or is not a file/directory: ${entry}" >&2
        exit 1
    fi
    local subdir
    subdir="$(find "${entry}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    if [[ -n "${subdir}" ]]; then
        echo "[checkpoint_accuracy_analysis.sh] strict checkpoint directory may not contain subdirectories: ${subdir}" >&2
        exit 1
    fi
    local non_pt
    non_pt="$(find -L "${entry}" -mindepth 1 -maxdepth 1 -type f ! -name '*.pt' -print -quit)"
    if [[ -n "${non_pt}" ]]; then
        echo "[checkpoint_accuracy_analysis.sh] strict checkpoint directory may contain .pt files only: ${non_pt}" >&2
        exit 1
    fi
    mapfile -t found_files < <(find -L "${entry}" -mindepth 1 -maxdepth 1 -type f -name '*.pt' -print | sort)
    if [[ ${#found_files[@]} -lt 1 ]]; then
        echo "[checkpoint_accuracy_analysis.sh] checkpoint directory contains no .pt files: ${entry}" >&2
        exit 1
    fi
    local file
    for file in "${found_files[@]}"; do
        printf '%s\n' "$(realpath "${file}")"
    done
}

checkpoint_files=()
for checkpoint_entry in "${CHECKPOINT_SET[@]}"; do
    checkpoint_entry="${checkpoint_entry:?checkpoint input is required}"
    while IFS= read -r resolved_file; do
        checkpoint_files+=("${resolved_file}")
    done < <(_collect_checkpoint_files "${checkpoint_entry}")
done

if [[ ${#checkpoint_files[@]} -lt 1 ]]; then
    echo '[checkpoint_accuracy_analysis.sh] no checkpoint files were discovered.' >&2
    exit 1
fi

if [[ "${CHECKPOINTS_PER_JOB}" == "all" ]]; then
    batch_size="${#checkpoint_files[@]}"
elif [[ "${CHECKPOINTS_PER_JOB}" =~ ^[0-9]+$ && "${CHECKPOINTS_PER_JOB}" -ge 1 ]]; then
    batch_size="${CHECKPOINTS_PER_JOB}"
else
    echo '[checkpoint_accuracy_analysis.sh] CHECKPOINTS_PER_JOB must be a positive integer or all.' >&2
    exit 1
fi

launch_count=0
batch_index=0
for ((start=0; start<${#checkpoint_files[@]}; start+=batch_size)); do
    batch_index=$((batch_index + 1))
    batch_files=("${checkpoint_files[@]:start:batch_size}")
    if [[ ${#batch_files[@]} -eq 1 ]]; then
        checkpoint_input="${batch_files[0]}"
        stem="$(basename "${checkpoint_input}" .pt)"
        case_id="$(_sanitize_token "${stem}")"
    else
        case_id="batch_$(printf '%04d' "${batch_index}")"
        checkpoint_input="${BATCH_DIR_ROOT}/${case_id}"
        mkdir -p "${checkpoint_input}"
        item_index=0
        for checkpoint_file in "${batch_files[@]}"; do
            item_index=$((item_index + 1))
            link_name="$(printf '%04d__%s' "${item_index}" "$(basename "${checkpoint_file}")")"
            ln -sf "${checkpoint_file}" "${checkpoint_input}/${link_name}"
        done
    fi
    gpu_index="${GPU_INDEX_SET[$(( (batch_index - 1) % ${#GPU_INDEX_SET[@]} ))]}"
    slot_id="${case_id}__gpu${gpu_index}"
    log_file="${LOG_DIR}/checkpoint_accuracy_analysis__${RUN_STAMP}__${slot_id}.log"
    cmd=("${PYTHON_BIN}" -m src.checkpoint_accuracy_analysis
        # Python 인수: --checkpoint
        # 기능: 단일 .pt file 또는 temporary .pt-only directory 를 전달한다.
        --checkpoint "${checkpoint_input}"
        # Python 인수: --dataset
        # 기능: checkpoint training dataset token 을 전달한다.
        --dataset "${DATASET}"
        # Python 인수: --prep_root
        # 기능: prepared dataset root 를 전달한다.
        --prep_root "${PREP_ROOT}"
        # Python 인수: --output_root
        # 기능: accuracy CSV 출력 root 를 전달한다.
        --output_root "${OUTPUT_ROOT}/${case_id}"
        # Python 인수: --anal_batch
        # 기능: evaluation batch size 를 전달한다.
        --anal_batch "${ANAL_BATCH}"
        # Python 인수: --gpu_index
        # 기능: CUDA device index 를 전달한다.
        --gpu_index "${gpu_index}"
        # Python 인수: --num_workers
        # 기능: DataLoader worker 수를 전달한다.
        --num_workers "${NUM_WORKERS}"
        # Python 인수: --splits
        # 기능: 평가할 split 목록을 전달한다.
        --splits "${SPLITS[@]}")
    if [[ -n "${SEED}" ]]; then
        # Python 인수: --seed
        # 기능: accuracy analysis seed 를 전달한다.
        cmd+=(--seed "${SEED}")
    fi
    printf -v command_text '%q ' "${cmd[@]}"
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN_STAMP}" "checkpoint_accuracy_analysis" "${case_id}" "${slot_id}" "$!" "${log_file}" "${command_text}" >> "${PID_FILE}"
    launch_count=$((launch_count + 1))
done

printf '[checkpoint_accuracy_analysis.sh] launched %d job(s); pid file: %s\n' "${launch_count}" "${PID_FILE}"
