#!/usr/bin/env bash
set -euo pipefail

# 기능: prepared dataset 으로 supervised model training 을 실행하고, 지정 epoch 의 .pt checkpoint 와 training metric CSV 를 저장한다.
# 실행 단위: BASE_SET x TRAINING_SETTING_SET x MODEL_SLOT_SET x READOUT_SET x REGULARIZATION_SET 의 Cartesian product 하나가 하나의 child job 이다.
# 실행 방식: 모든 training case 를 nohup background process 로 즉시 실행하고, parent launcher 는 종료를 기다리지 않는다.
# 산출물: checkpoint 는 `<OUTPUT_ROOT>/checkpoints/<case_id>` 에, metric CSV 는 `<OUTPUT_ROOT>/metrics/<case_id>` 에 저장된다.
# 주의: 이 launcher 는 signal analysis 나 plotting 을 실행하지 않는다.
# 로그: `<LOG_ROOT>/model_training/<RUN_STAMP>` 아래 job별 log 와 pid.tsv 를 생성한다.

# 인수: PROJECT_ROOT
# 기능: 이 저장소의 최상위 경로를 지정한다.
# 줄 수 있는 값: 절대경로 또는 현재 작업 위치 기준 상대경로. 기본값은 이 스크립트의 부모 디렉터리이다.
# 값의 의미: Python module 실행 시 src/ 를 import 할 기준 경로이다.
# 값 주는법: `PROJECT_ROOT=/absolute/path/to/psd bash bash/model_training.sh` 처럼 준다.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 인수: PYTHON_BIN
# 기능: child job 을 실행할 Python executable 을 지정한다.
# 줄 수 있는 값: `python3`, `python`, conda 환경의 `/path/to/python` 같은 실행 가능 파일.
# 값의 의미: 지정한 Python 으로 `src.model_training` module 을 실행한다.
# 값 주는법: `PYTHON_BIN=/home/user/miniconda3/envs/snn/bin/python bash bash/model_training.sh` 처럼 준다.
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 인수: PREP_ROOT
# 기능: prepared dataset bundle 을 읽을 root directory 를 지정한다.
# 줄 수 있는 값: 절대경로. 기본값은 `/home/yongokhan/바탕화면/prep_data` 이다.
# 값의 의미: 입력 prepared dataset path 는 `<PREP_ROOT>/<dataset>` 이다.
# 값 주는법: `PREP_ROOT=/absolute/path/to/prep_data` 처럼 준다.
PREP_ROOT="${PREP_ROOT:-/home/yongokhan/바탕화면/prep_data}"

# 인수: OUTPUT_ROOT
# 기능: checkpoint 와 metric CSV 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: checkpoint root 와 metric root 의 부모 directory 이다.
# 값 주는법: `OUTPUT_ROOT=/absolute/path/to/outputs` 처럼 준다.
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

# 인수: LOG_ROOT
# 기능: launcher log 와 pid.tsv 를 저장할 root directory 를 지정한다.
# 줄 수 있는 값: 반드시 제공해야 하는 경로 문자열.
# 값의 의미: 실제 log directory 는 `<LOG_ROOT>/model_training/<RUN_STAMP>` 이다.
# 값 주는법: `LOG_ROOT=/absolute/path/to/logs` 처럼 준다.
LOG_ROOT="${LOG_ROOT:-/home/yongokhan/바탕화면/logs}"

# 인수: RUN_STAMP
# 기능: 한 번 실행한 launcher 묶음을 구분하는 실행 표식을 지정한다.
# 줄 수 있는 값: 파일명에 안전한 문자열. 기본값은 `YYYYmmdd_HHMMSS` 형식의 현재 시각이다.
# 값의 의미: 같은 RUN_STAMP 를 쓰면 같은 실행 묶음의 log directory 에 기록된다.
# 값 주는법: `RUN_STAMP=20260430_120000` 처럼 준다.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# 인수: EPOCHS
# 기능: TRAINING_SETTING_SET 을 따로 주지 않았을 때 사용할 기본 training epoch 수이다.
# 줄 수 있는 값: 양의 정수. 기본값은 `1` 이다.
# 값의 의미: 전체 dataset 을 몇 epoch 학습할지 정한다.
# 값 주는법: `EPOCHS=25` 처럼 준다.
EPOCHS="${EPOCHS:-50}"

# 인수: BATCH_SIZE
# 기능: TRAINING_SETTING_SET 을 따로 주지 않았을 때 사용할 기본 training batch size 이다.
# 줄 수 있는 값: 양의 정수. 기본값은 `64` 이다.
# 값의 의미: 한 optimizer step 에 들어가는 sample 수이다.
# 값 주는법: `BATCH_SIZE=128` 처럼 준다.
BATCH_SIZE="${BATCH_SIZE:-256}"

# 인수: LR
# 기능: TRAINING_SETTING_SET 을 따로 주지 않았을 때 사용할 기본 learning rate 이다.
# 줄 수 있는 값: 양의 실수. 기본값은 `1e-3` 이다.
# 값의 의미: optimizer step 크기이다.
# 값 주는법: `LR=5e-4` 처럼 준다.
LR="${LR:-0.005}"

# 인수: NUM_WORKERS
# 기능: DataLoader worker process 수를 지정한다.
# 줄 수 있는 값: 0 이상의 정수. 0이면 기존처럼 main process에서 데이터를 읽는다.
# 값의 의미: 4 정도를 주면 CPU가 batch를 미리 준비해서 GPU 대기 시간을 줄일 수 있다.
# 값 주는법: `NUM_WORKERS=4` 처럼 준다.
NUM_WORKERS="${NUM_WORKERS:-4}"

# 인수: SEED
# 기능: TRAINING_SETTING_SET 을 따로 주지 않았을 때 사용할 기본 seed 이다.
# 줄 수 있는 값: 정수. 기본값은 `0` 이다.
# 값의 의미: model initialization, data order, checkpoint metadata 의 seed 로 사용된다.
# 값 주는법: `SEED=0` 처럼 준다.
SEED="${SEED:-0}"

# 인수: BASE_SET
# 기능: dataset 과 hidden layout 조합 목록을 지정한다.
# 줄 수 있는 값: 공백으로 구분한 `<dataset_token>|<hidden_spec>` 목록. hidden_spec 자체가 콤마를 쓸 수 있으므로 BASE_SET 원소 사이에는 공백을 사용한다.
# 값의 모든 dataset 선택지: `cifar-10`, `cifar10-dvs`, `deap`, `dvs128-gesture`, `mnist`, `n-mnist`, `ps-mnist`, `s-cifar10`, `s-mnist`, `shd`, `ssc`.
# 값의 모든 hidden_spec 형식: dense SNN 은 `128`, `256,256`, `64,32,20` 같은 양의 정수 CSV 를 쓴다. fixed CNN token 은 `-`, 빈 문자열, `default` 중 하나를 no-tail-width 의미로 쓴다.
# 값의 의미: `s-mnist|128` 은 s-mnist 에 hidden width 128 layout 을 쓰겠다는 뜻이다.
# 값 주는법: `BASE_SET="s-mnist|128 shd|256,256"` 처럼 준다.
BASE_SET_RAW="${BASE_SET:-s-mnist|128,64,64}"

# 인수: MODEL_SLOT_SET
# 기능: model token 과 GPU 조합 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 `<model_token>|<gpu_index>` 목록.
# 값의 모든 dense model token 형식: `lif_<soft|hard>_<fixed|train>`, `lif_R_<soft|hard>_<fixed|train>`, `rf_<soft|hard>_<fixed|train>`, `rf_R_<soft|hard>_<fixed|train>`.
# 값의 모든 fixed CNN token 형식: `<vgg11|resnet18>_<lif|rf>_<soft|hard>_<fixed|train>`.
# 값의 모든 auxiliary token: `spikingssm`, `spikformer`, `spikegru`.
# 값의 모든 gpu_index 선택지: CUDA 가 있으면 device 범위 안의 0 이상의 정수이다. CUDA 가 없으면 Python training code 가 CPU 로 실행한다.
# 값의 의미: `lif_soft_fixed|0` 은 해당 model 을 cuda:0 에 배정한다는 뜻이다.
# 값 주는법: `MODEL_SLOT_SET="lif_soft_fixed|0 rf_R_soft_fixed|1"` 처럼 준다.
MODEL_SLOT_SET_RAW="${MODEL_SLOT_SET:-lif_soft_fixed|0}"

# 인수: READOUT_SET
# 기능: output readout 방식 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 `final_membrane`, `first_spike`, `max_rate`.
# 값의 모든 의미: `final_membrane` 은 마지막 membrane 기반, `first_spike` 는 최초 spike timing 기반, `max_rate` 는 spike/rate 최대 집계 기반이다.
# 값 주는법: `READOUT_SET="final_membrane first_spike"` 또는 `READOUT_SET="final_membrane,first_spike"` 처럼 준다.
READOUT_SET_RAW="${READOUT_SET:-temporal_membrane}"

# 인수: TRAINING_SETTING_SET
# 기능: epoch, batch size, learning rate, seed 조합 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 `<epochs>|<batch_size>|<lr>|<seed>` 목록.
# 값의 모든 field 의미: epochs 는 양의 정수, batch_size 는 양의 정수, lr 은 양의 실수, seed 는 정수이다.
# 값의 의미: `25|128|1e-3|0` 은 25 epoch, batch 128, learning rate 0.001, seed 0 을 뜻한다.
# 값 주는법: `TRAINING_SETTING_SET="25|128|1e-3|0 50|128|1e-3|1"` 처럼 준다. 생략하면 EPOCHS, BATCH_SIZE, LR, SEED 로 만든 단일 조합을 쓴다.
TRAINING_SETTING_SET_RAW="${TRAINING_SETTING_SET:-${EPOCHS}|${BATCH_SIZE}|${LR}|${SEED}}"

# 인수: ANAL_EPOCH_LIST
# 기능: 나중에 psd_analysis 에 사용할 checkpoint 저장 epoch 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 정수 목록. 각 값은 `1 <= epoch <= EPOCHS` 범위여야 한다.
# 값의 의미: 목록에 포함된 epoch 만 .pt checkpoint 로 저장된다. 비우면 final epoch 만 저장한다.
# 값 주는법: `ANAL_EPOCH_LIST="5 10 25"` 또는 `ANAL_EPOCH_LIST="5,10,25"` 처럼 준다.
ANAL_EPOCH_LIST_RAW="${ANAL_EPOCH_LIST:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50}"

# 인수: REGULARIZATION_SET
# 기능: model_training 단계에서 supervised task loss 에 더할 PSD curve-shape training-side regularization 조합 목록을 지정한다.
# 줄 수 있는 값: 공백 또는 콤마로 구분한 `<lambda1>|<lambda2>|<signal>|<curve_space>|<curve_scale>|<centering>|<reducer>` 목록. 거리 지표는 REGULARIZATION_DISTANCE_METRIC 으로 전체 실행에 하나만 지정한다.
# 값의 모든 lambda1 선택지: 0 이상의 실수이다. `0` 은 입력곡선-모든 히든레이어곡선 거리합 항을 끈다.
# 값의 모든 lambda2 선택지: 0 이상의 실수이다. `0` 은 입력-첫 히든 및 히든-히든 인접곡선 거리합 항을 끈다.
# 값의 모든 signal 선택지: `y_mem`, `y_spike`. 한 조합 안에서는 이 한 curve family 만 고정해서 모든 히든레이어에 사용한다.
# 값의 모든 curve_space 선택지: `exact`.
# 값의 모든 curve_scale 선택지: `raw`, `db`.
# 값의 모든 centering 선택지: `raw`, `centered`.
# 값의 모든 reducer 선택지: `mean`, `median`.
# 값의 의미: `1e-4|1e-5|y_mem|exact|raw|centered|mean` 은 signal.md 의 d_shape 로 lambda1 * sum d_shape(입력곡선, 히든레이어곡선) 과 lambda2 * sum d_shape(입력-첫 히든 및 히든-히든 인접곡선)를 total loss 에 더한다.
# 값 주는법: `REGULARIZATION_SET="0|0|y_mem|exact|raw|raw|mean 1e-4|1e-5|y_spike|exact|db|centered|median"` 처럼 준다.
REGULARIZATION_SET_RAW="0|0.001|y_spike|exact|db|raw|mean 0|0|y_spike|exact|db|raw|mean 0|-0.001|y_spike|exact|db|raw|mean}"

# 인수: REGULARIZATION_DISTANCE_METRIC
# 기능: PSD curve-shape regularization 에 사용할 거리 지표를 지정한다.
# 줄 수 있는 값: `centered_l2`, `diff_l2`. 기본값은 기존 동작 보존을 위해 `centered_l2` 이다.
# 값의 의미: `centered_l2` 는 기존 평균 제거 후 L2, `diff_l2` 는 정규화 없는 1차 차분 L2 이다.
# 값 주는법: `REGULARIZATION_DISTANCE_METRIC=diff_l2 bash bash/model_training.sh` 처럼 준다.
REGULARIZATION_DISTANCE_METRIC="${REGULARIZATION_DISTANCE_METRIC:-centered_l2}"


# 기능: log file name 과 pid.tsv key 에 안전하게 쓸 수 있도록 token 을 정규화한다.
_sanitize_token() {
    local raw="$1"
    printf '%s' "${raw}" | tr '/:|, ' '____' | tr -cs '[:alnum:]_.=-' '_'
}

# 기능: pipe 기반 scenario 원소가 지정된 field 개수와 일치하는지 검사한다.
_assert_pipe_field_count() {
    local label="$1" item="$2" expected="$3" actual
    actual="$(awk -F'|' '{print NF}' <<< "${item}")"
    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "[model_training.sh] ${label} entries must contain exactly ${expected} pipe-separated fields: ${item}" >&2
        exit 1
    fi
}

LOG_DIR="${LOG_ROOT}/model_training/${RUN_STAMP}"
mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/metrics" "${LOG_DIR}"
PID_FILE="${LOG_DIR}/pid.tsv"
printf 'run_stamp\tstage\tcase_id\tslot_id\tpid\tlog_path\tcommand\n' > "${PID_FILE}"

# BASE_SET 은 hidden_spec 에 콤마가 들어갈 수 있으므로 콤마를 원소 구분자로 쓰지 않는다.
MODEL_SLOT_SET_NORMALIZED="${MODEL_SLOT_SET_RAW//,/ }"
READOUT_SET_NORMALIZED="${READOUT_SET_RAW//,/ }"
TRAINING_SETTING_SET_NORMALIZED="${TRAINING_SETTING_SET_RAW//,/ }"
ANAL_EPOCH_LIST_NORMALIZED="${ANAL_EPOCH_LIST_RAW//,/ }"
REGULARIZATION_SET_NORMALIZED="${REGULARIZATION_SET_RAW//,/ }"
IFS=' ' read -r -a BASE_SET <<< "${BASE_SET_RAW}"
IFS=' ' read -r -a MODEL_SLOT_SET <<< "${MODEL_SLOT_SET_NORMALIZED}"
IFS=' ' read -r -a READOUT_SET <<< "${READOUT_SET_NORMALIZED}"
IFS=' ' read -r -a TRAINING_SETTING_SET <<< "${TRAINING_SETTING_SET_NORMALIZED}"
IFS=' ' read -r -a ANAL_EPOCH_LIST <<< "${ANAL_EPOCH_LIST_NORMALIZED}"
IFS=' ' read -r -a REGULARIZATION_SET <<< "${REGULARIZATION_SET_NORMALIZED}"

anal_epoch_args=()
if [[ ${#ANAL_EPOCH_LIST[@]} -gt 0 && -n "${ANAL_EPOCH_LIST[0]:-}" ]]; then
    anal_epoch_args=(--anal_epoch_list "${ANAL_EPOCH_LIST[@]}")
fi


launch_count=0
for base in "${BASE_SET[@]}"; do
    _assert_pipe_field_count BASE_SET "${base}" 2
    IFS='|' read -r dataset_token hidden_spec <<< "${base}"
    for train_setting in "${TRAINING_SETTING_SET[@]}"; do
        _assert_pipe_field_count TRAINING_SETTING_SET "${train_setting}" 4
        IFS='|' read -r epochs batch_size lr seed <<< "${train_setting}"
        for model_slot in "${MODEL_SLOT_SET[@]}"; do
            _assert_pipe_field_count MODEL_SLOT_SET "${model_slot}" 2
            IFS='|' read -r model_token gpu_index <<< "${model_slot}"
            for readout_mode in "${READOUT_SET[@]}"; do
                for regularization_setting in "${REGULARIZATION_SET[@]}"; do
                    _assert_pipe_field_count REGULARIZATION_SET "${regularization_setting}" 7
                    IFS='|' read -r regularization_lambda1 regularization_lambda2 regularization_signal regularization_curve_space regularization_curve_scale regularization_centering regularization_reducer <<< "${regularization_setting}"
                    reg_case="$(_sanitize_token "reg_l1${regularization_lambda1}__l2${regularization_lambda2}__sig${regularization_signal}__space${regularization_curve_space}__scale${regularization_curve_scale}__center${regularization_centering}__red${regularization_reducer}__dist${REGULARIZATION_DISTANCE_METRIC}")"
                    case_id="$(_sanitize_token "${dataset_token}__${model_token}__${hidden_spec}__${readout_mode}__e${epochs}__b${batch_size}__lr${lr}__seed${seed}__${reg_case}")"
                    slot_id="${case_id}__gpu${gpu_index}"
                    checkpoint_root="${OUTPUT_ROOT}/checkpoints/${case_id}"
                    metric_root="${OUTPUT_ROOT}/metrics/${case_id}"
                    log_path="${LOG_DIR}/model_training__${RUN_STAMP}__${slot_id}.log"
                    cmd=("${PYTHON_BIN}" -m src.model_training
                        # Python 인수: --dataset
                        # 기능: training 대상 dataset token 을 전달한다.
                        # 줄 수 있는 값: BASE_SET 의 dataset_token field.
                        # 값의 의미: `<PREP_ROOT>/<dataset>` prepared bundle 을 읽는다.
                        --dataset "${dataset_token}"
                        # Python 인수: --prep_root
                        # 기능: prepared dataset root 를 전달한다.
                        # 줄 수 있는 값: 절대경로.
                        # 값의 의미: prepared dataset path 는 `<prep_root>/<dataset>` 이다.
                        --prep_root "${PREP_ROOT}"
                        # Python 인수: --model
                        # 기능: model token 을 전달한다.
                        # 줄 수 있는 값: MODEL_SLOT_SET 설명의 model_token 형식.
                        # 값의 의미: SNN model family, recurrent 여부, reset, threshold, backbone 을 결정한다.
                        --model "${model_token}"
                        # Python 인수: --hidden_spec
                        # 기능: hidden layer layout 또는 fixed CNN placeholder 를 전달한다.
                        # 줄 수 있는 값: BASE_SET 설명의 hidden_spec 형식.
                        # 값의 의미: dense model 에서는 hidden widths, fixed CNN 에서는 no-tail-width 의미이다.
                        --hidden_spec "${hidden_spec}"
                        # Python 인수: --readout_mode
                        # 기능: readout 방식을 전달한다.
                        # 줄 수 있는 값: `final_membrane`, `first_spike`, `max_rate`.
                        # 값의 의미: loss 와 prediction 을 만들 output 해석 방식을 결정한다.
                        --readout_mode "${readout_mode}"
                        # Python 인수: --epochs
                        # 기능: 학습 epoch 수를 전달한다.
                        # 줄 수 있는 값: 양의 정수.
                        # 값의 의미: supervised training 반복 횟수이다.
                        --epochs "${epochs}"
                        # Python 인수: --batch_size
                        # 기능: training batch size 를 전달한다.
                        # 줄 수 있는 값: 양의 정수.
                        # 값의 의미: optimizer step 당 sample 수이다.
                        --batch_size "${batch_size}"
                        # Python 인수: --num_workers
                        # 기능: DataLoader worker process 수를 전달한다.
                        # 줄 수 있는 값: 0 이상의 정수.
                        # 값의 의미: batch loading/preparation 병렬도를 정한다.
                        --num_workers "${NUM_WORKERS}"
                        # Python 인수: --lr
                        # 기능: learning rate 를 전달한다.
                        # 줄 수 있는 값: 양의 실수.
                        # 값의 의미: optimizer update step 크기이다.
                        --lr "${lr}"
                        # Python 인수: --seed
                        # 기능: training seed 를 전달한다.
                        # 줄 수 있는 값: 정수.
                        # 값의 의미: 초기화, data order, checkpoint metadata 에 쓰인다.
                        --seed "${seed}"
                        # Python 인수: --gpu_index
                        # 기능: GPU index 를 전달한다.
                        # 줄 수 있는 값: CUDA 가 있으면 device 범위 안의 0 이상의 정수이다.
                        # 값의 의미: training 에 사용할 device index 이다.
                        --gpu_index "${gpu_index}"
                        # Python 인수: --regularization_lambda1
                        # 기능: 입력곡선 C0 와 모든 히든레이어곡선 Ci 사이의 d_shape 거리합에 곱할 계수를 전달한다.
                        # 줄 수 있는 값: 0 이상의 실수.
                        # 값의 의미: total loss 에 lambda1 * sum_i d_shape(C0, Ci) 를 더한다. `0` 이면 이 입력-히든 전체 항을 사용하지 않는다.
                        --regularization_lambda1 "${regularization_lambda1}"
                        # Python 인수: --regularization_lambda2
                        # 기능: 입력-첫 히든 및 히든-히든 인접곡선 쌍의 d_shape 거리합에 곱할 계수를 전달한다.
                        # 줄 수 있는 값: 0 이상의 실수.
                        # 값의 의미: total loss 에 lambda2 * sum_i d_shape(C_{i-1}, Ci) 를 더한다. 인접쌍은 입력-첫 히든을 포함하고 마지막 히든-출력 쌍은 포함하지 않는다. `0` 이면 이 인접 항을 사용하지 않는다.
                        --regularization_lambda2 "${regularization_lambda2}"
                        # Python 인수: --regularization_signal
                        # 기능: PSD 기반 규제항 계산에 사용할 히든레이어 curve family 를 전달한다.
                        # 줄 수 있는 값: `y_mem`, `y_spike`.
                        # 값의 의미: 모든 히든레이어에서 같은 family 를 사용해 PSD curve 를 만든다.
                        --regularization_signal "${regularization_signal}"
                        # Python 인수: --regularization_curve_space
                        # 기능: 규제항 PSD curve 의 frequency-axis representation 을 전달한다.
                        # 줄 수 있는 값: `exact`.
                        # 값의 의미: `exact` 는 rFFT one-sided grid 이다.
                        --regularization_curve_space "${regularization_curve_space}"
                        # Python 인수: --regularization_curve_scale
                        # 기능: 규제항 PSD curve 의 값 scale 을 전달한다.
                        # 줄 수 있는 값: `raw`, `db`.
                        # 값의 의미: `raw` 는 power, `db` 는 10log10 power scale 이다.
                        --regularization_curve_scale "${regularization_curve_scale}"
                        # Python 인수: --regularization_centering
                        # 기능: PSD 계산 전에 signal 의 time mean 제거 여부를 전달한다.
                        # 줄 수 있는 값: `raw`, `centered`.
                        # 값의 의미: `raw` 는 원 신호, `centered` 는 시간축 평균을 제거한 신호이다.
                        --regularization_centering "${regularization_centering}"
                        # Python 인수: --regularization_reducer
                        # 기능: PSD map 의 row axis 를 curve 로 줄이는 방식을 전달한다.
                        # 줄 수 있는 값: `mean`, `median`.
                        # 값의 의미: 각 sample 의 row-wise PSD 를 평균 또는 median 으로 대표 curve 로 만든다.
                        --regularization_reducer "${regularization_reducer}"
                        # Python 인수: --regularization_distance_metric
                        # 기능: PSD curve-shape regularization 의 d_shape 거리 지표를 전달한다.
                        # 줄 수 있는 값: `centered_l2`, `diff_l2`.
                        # 값의 의미: `centered_l2` 는 기존 평균 제거 후 L2, `diff_l2` 는 정규화 없는 1차 차분 L2 이다.
                        --regularization_distance_metric "${REGULARIZATION_DISTANCE_METRIC}"
                        # Python 인수: --checkpoint_root
                        # 기능: .pt checkpoint 저장 directory 를 전달한다.
                        # 줄 수 있는 값: 경로 문자열.
                        # 값의 의미: 이 directory 에는 selected .pt checkpoint file 만 저장한다.
                        --checkpoint_root "${checkpoint_root}"
                        # Python 인수: --metric_root
                        # 기능: training metric CSV 저장 directory 를 전달한다.
                        # 줄 수 있는 값: 경로 문자열.
                        # 값의 의미: checkpoint directory 밖에 metric CSV 를 저장한다.
                        --metric_root "${metric_root}"
                        # Python 인수: --output_root
                        # 기능: parent output root 를 metadata 용으로 전달한다.
                        # 줄 수 있는 값: 경로 문자열.
                        # 값의 의미: 실행 산출물의 상위 root 정보를 기록한다.
                        --output_root "${OUTPUT_ROOT}"
                        # Python 인수: --anal_epoch_list
                        # 기능: checkpoint 저장 epoch 목록을 선택적으로 전달한다.
                        # 줄 수 있는 값: ANAL_EPOCH_LIST 설명의 정수 목록.
                        # 값의 의미: 전달된 epoch 만 selected checkpoint 로 저장한다.
                        "${anal_epoch_args[@]}")
                    printf -v command_text '%q ' "${cmd[@]}"
                    nohup "${cmd[@]}" > "${log_path}" 2>&1 &
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${RUN_STAMP}" "model_training" "${case_id}" "${slot_id}" "$!" "${log_path}" "${command_text}" >> "${PID_FILE}"
                    launch_count=$((launch_count + 1))
                done
            done
        done
    done
done

printf '[model_training.sh] launched %d job(s); pid file: %s\n' "${launch_count}" "${PID_FILE}"
