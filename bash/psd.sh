#!/usr/bin/env bash
set -euo pipefail

: "${ROOT_DIR:?ROOT_DIR must be set by bash/run_psd.sh}"
: "${PSD_RUN_LAUNCHER_ACTIVE:?Use bash/run_psd.sh. bash/psd.sh is an internal config template and is not meant to be executed directly.}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# -----------------------------------------------------------------------------
# 역할
# -----------------------------------------------------------------------------
# - 이 파일은 psd_analysis 실험의 Python / ML 쪽 기본 인수 template 이다.
# - 직접 nohup 실행하는 launcher 가 아니라, bash/run_psd.sh 가 내부적으로 호출하는 config script 다.
# - 값을 비워 두면 해당 CLI 인수를 생략한다.
# - required 인수를 여기서 비워 둔 경우 run_psd.sh 의 scenario override 또는 최종 CLI pass-through 로 채워야 한다.
# - output layer 뒤 learned NN head 는 두지 않는다.
# - readout 은 output neuron 의 membrane / spike sequence 에 대한 기능적 판정 규칙이다.
# - DATASET 은 단일 값만 허용한다.
# - MODELS / READOUT_MODES 에 여러 값을 주면 psd_analysis.py 가 model × readout 조합을 받은 순서대로 직렬 실행한다.
# - grouped model / readout 병렬 시나리오는 bash/run_psd.sh 가 담당한다.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 공식 dataset / model / readout 규칙
# -----------------------------------------------------------------------------
# dataset canonical name
#   s-mnist / dvsgesture / shd / deap / forda
#
# model token 예시
#   lif, lif_R, lif_struct, lif_struct_R, lif_clip, lif_structclip
#   rf, rf_R, rf_struct, rf_struct_R, rf_clip, rf_structclip
#   tc, tc_lif, tc_lif_R
#   ts, ts_lif, ts_lif_R
#   dh, dh_snn, dh_snn_R, dh_snn_R_4
#   d_rf, d_rf_4
#   my_dh_snn, my_r_dh_snn, my_d_rf
#
# readout mode
#   final_membrane / first_spike / max_rate
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PSD / 저장 규칙 요약
# -----------------------------------------------------------------------------
# - dataset 입력 probe reference 저장은 dataset_psd 가 담당한다.
# - psd_analysis 는 선택된 epoch 의 hidden/output bundle 과 부가 통계를 저장한다.
# - hidden/output signal bundle 은 raw / centered 선형 8개 PNG + 대응 dB 8개 PNG + summary.json 을 저장한다.
# - 선택된 epoch root 에는 probe_set_accuracy.txt, attenuation_stats/, all_layers_summary.csv,
#   hidden-layer w_plot.png, grouped 시나리오의 block PSD bundle / block weight plot 이 저장된다.
# - 각 selected epoch 의 각 layer / family 디렉터리에는 시간영역 플롯 두 개를 추가로 저장한다.
#   * time_domain_heatmap.png : x축 timestep, y축 element index
#   * time_domain_element_mean.png : x축 timestep, y축 element 평균
# - 최종 snapshot 은 training_complete_stats/ 아래에 최종 probe-set accuracy,
#   최종 attenuation 통계, 최종 all_layers_summary.csv, 최종 centered pointwise L2 semi-metric(디렉터리명은 shape_sim_metric) 을 유지한다.
# - 학습 중에는 PNG 를 직접 그리지 않고 process-local CPU 메모리에 numeric payload 만 홀드한 뒤,
#   학습 완료 후 tqdm 로그와 함께 렌더링하며 성공한 payload 는 즉시 메모리에서 제거한다.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 배열 인수 표기 규칙 (중요)
# -----------------------------------------------------------------------------
# - DATASET 은 단일 scalar 값이다.
# - MODELS, READOUT_MODES, HIDDEN, PLOT_EPOCHS, USERBIN_EDGES, W_CLIP_EDGES,
#   ALPHA_CLIP_EDGES, BAND_NEURON_ENDS 는 공백 분리 bash 배열이다.
# - 빈 배열이면 해당 CLI 인수를 전달하지 않는다.
# - required 인수(`--dataset`, `--model`, `--hidden`)를 여기서 비워 두면 반드시
#   bash/run_psd.sh 나 최종 CLI pass-through 로 채워야 한다.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 기본 경로 / 실행 대상 (비우면 생략)
# -----------------------------------------------------------------------------
DATA_ROOT=""        # --data_root : 절대경로. 비우면 생략
OUT_ROOT=""         # --out_root  : 절대경로. 비우면 생략

# -----------------------------------------------------------------------------
# CUDA / seed / dataloader (비우면 엔트리포인트 기본값 사용)
# -----------------------------------------------------------------------------
GPU=""              # --gpu : CUDA device index. 비우면 생략
SEED=""             # --seed : 전역 시드 + deterministic probe-set 선택 시드
NUM_WORKERS=""      # --num_workers : PyTorch DataLoader worker 수
DOWNLOAD=""         # --download : 0 또는 1. 비우면 생략

# -----------------------------------------------------------------------------
# dataset / model / readout / hidden (required set; 여기서는 빈 template)
# -----------------------------------------------------------------------------
DATASET=""          # --dataset : 예) s-mnist
MODELS=()            # --model : 예) (lif lif_R rf dh_snn_R_4)
READOUT_MODES=()     # --readout_mode : 예) (final_membrane first_spike)
HIDDEN=()            # --hidden : 예) (256 256 128)

# -----------------------------------------------------------------------------
# 학습 하이퍼파라미터 (비우면 생략 -> 엔트리포인트 기본값 사용)
# -----------------------------------------------------------------------------
EPOCHS=""
BATCH_SIZE=""
LR=""
WEIGHT_DECAY=""
WEIGHT_DECAY_DEND_SOMA=""
SOFT_MASK_EPOCHS=""
STABILIZE_EPOCHS=""
STE_EPOCHS=""

# -----------------------------------------------------------------------------
# 공통 neuron / model 하이퍼파라미터 (비우면 생략)
# -----------------------------------------------------------------------------
S_MIN=""
S_MAX=""
TH_LEN=""
V_TH=""
V_PRE=""
LAMBDA_ORTHO=""
LAMBDA_S=""

# -----------------------------------------------------------------------------
# probe set / PSD / spectrogram 설정
# -----------------------------------------------------------------------------
SAME_LABEL_N_PER_LABEL=""
BALANCED_GLOBAL_N_PER_LABEL=""
PLOT_EPOCHS=()
PSD_WINDOW=""
PSD_OVERLAP=""
USERBIN_EDGES=()

# -----------------------------------------------------------------------------
# SHD 전용 인수
# -----------------------------------------------------------------------------
SHD_T=""
SHD_MAX_TIME=""
SHD_BINNING=""
SHD_UNIT_INDEXING=""
SHD_CHANNEL_FLIP=""
SHD_ALIGN_TO_FIRST_EVENT=""
SHD_USE_EVENT_COUNTS=""

# -----------------------------------------------------------------------------
# DVS128 Gesture 전용 인수
# -----------------------------------------------------------------------------
DVSGESTURE_CHUNK_SIZE=""
DVSGESTURE_EMPTY_SIZE=""
DVSGESTURE_DT_MS=""
DVSGESTURE_DS=""

# -----------------------------------------------------------------------------
# DEAP 전용 인수
# -----------------------------------------------------------------------------
DEAP_LABEL_AXIS=""
DEAP_NUM_CLASSES=""

# -----------------------------------------------------------------------------
# RF / LIF / 구조 분리 전용 인수
# -----------------------------------------------------------------------------
RF_RESET_MODE=""
W_CLIP_EDGES=()
ALPHA_CLIP_EDGES=()
BAND_NEURON_ENDS=()
TEAR=""

# -----------------------------------------------------------------------------
# run naming
# -----------------------------------------------------------------------------
EXP_NAME=""
CUSTOM_TIMESTAMP=""

# Append one scalar CLI option only when the value is non-empty.
#
# This lets the template keep many knobs blank while still producing a clean
# final ``psd_analysis.py`` command line.
append_scalar_if_nonempty() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    CMD+=("${flag}" "${value}")
  fi
}

# Append one multi-value CLI option only when the array contains entries.
#
# Model, readout, hidden-width, and clip-edge arguments all use bash arrays here
# because the Python CLI expects repeated positional values after a single flag.
append_array_if_nonempty() {
  local flag="$1"
  local -n values_ref="$2"
  if [[ ${#values_ref[@]} -gt 0 ]]; then
    CMD+=("${flag}" "${values_ref[@]}")
  fi
}

CMD=("${PYTHON_BIN}" -u "${ROOT_DIR}/src/psd_analysis.py")

append_scalar_if_nonempty --dataset "${DATASET}"
append_array_if_nonempty --model MODELS
append_array_if_nonempty --readout_mode READOUT_MODES
append_scalar_if_nonempty --out_root "${OUT_ROOT}"
append_scalar_if_nonempty --data_root "${DATA_ROOT}"
append_scalar_if_nonempty --exp_name "${EXP_NAME}"
append_scalar_if_nonempty --timestamp "${CUSTOM_TIMESTAMP}"
append_scalar_if_nonempty --gpu "${GPU}"
append_scalar_if_nonempty --seed "${SEED}"
append_array_if_nonempty --hidden HIDDEN
append_scalar_if_nonempty --epochs "${EPOCHS}"
append_scalar_if_nonempty --soft_mask_epochs "${SOFT_MASK_EPOCHS}"
append_scalar_if_nonempty --stabilize_epochs "${STABILIZE_EPOCHS}"
append_scalar_if_nonempty --ste_epochs "${STE_EPOCHS}"
append_scalar_if_nonempty --batch_size "${BATCH_SIZE}"
append_scalar_if_nonempty --lr "${LR}"
append_scalar_if_nonempty --weight_decay "${WEIGHT_DECAY}"
append_scalar_if_nonempty --weight_decay_dend_soma "${WEIGHT_DECAY_DEND_SOMA}"
append_scalar_if_nonempty --S_min "${S_MIN}"
append_scalar_if_nonempty --S_max "${S_MAX}"
append_scalar_if_nonempty --th_len "${TH_LEN}"
append_scalar_if_nonempty --v_th "${V_TH}"
append_scalar_if_nonempty --v_pre "${V_PRE}"
append_scalar_if_nonempty --num_workers "${NUM_WORKERS}"
append_scalar_if_nonempty --download "${DOWNLOAD}"
append_scalar_if_nonempty --shd_T "${SHD_T}"
append_scalar_if_nonempty --shd_max_time "${SHD_MAX_TIME}"
append_scalar_if_nonempty --shd_binning "${SHD_BINNING}"
append_scalar_if_nonempty --shd_unit_indexing "${SHD_UNIT_INDEXING}"
append_scalar_if_nonempty --shd_channel_flip "${SHD_CHANNEL_FLIP}"
append_scalar_if_nonempty --shd_align_to_first_event "${SHD_ALIGN_TO_FIRST_EVENT}"
append_scalar_if_nonempty --shd_use_event_counts "${SHD_USE_EVENT_COUNTS}"
append_scalar_if_nonempty --dvsgesture_chunk_size "${DVSGESTURE_CHUNK_SIZE}"
append_scalar_if_nonempty --dvsgesture_empty_size "${DVSGESTURE_EMPTY_SIZE}"
append_scalar_if_nonempty --dvsgesture_dt_ms "${DVSGESTURE_DT_MS}"
append_scalar_if_nonempty --dvsgesture_ds "${DVSGESTURE_DS}"
append_scalar_if_nonempty --deap_label_axis "${DEAP_LABEL_AXIS}"
append_scalar_if_nonempty --deap_num_classes "${DEAP_NUM_CLASSES}"
append_scalar_if_nonempty --lambda_ortho "${LAMBDA_ORTHO}"
append_scalar_if_nonempty --lambda_s "${LAMBDA_S}"
append_scalar_if_nonempty --same_label_n_per_label "${SAME_LABEL_N_PER_LABEL}"
append_scalar_if_nonempty --balanced_global_n_per_label "${BALANCED_GLOBAL_N_PER_LABEL}"
append_array_if_nonempty --plot_epoch PLOT_EPOCHS
append_scalar_if_nonempty --psd_window "${PSD_WINDOW}"
append_scalar_if_nonempty --psd_overlap "${PSD_OVERLAP}"
append_array_if_nonempty --userbin_edges USERBIN_EDGES
append_scalar_if_nonempty --rf_reset_mode "${RF_RESET_MODE}"
append_array_if_nonempty --w_clip_edges W_CLIP_EDGES
append_array_if_nonempty --alpha_clip_edges ALPHA_CLIP_EDGES
append_array_if_nonempty --band_neuron_ends BAND_NEURON_ENDS
append_scalar_if_nonempty --tear "${TEAR}"

CMD+=("$@")
exec "${CMD[@]}"
