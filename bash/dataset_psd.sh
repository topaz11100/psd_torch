#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
_INTERNAL_FLAG="__dataset_psd_internal__"

# -----------------------------------------------------------------------------
# 개요
# -----------------------------------------------------------------------------
# - 이 스크립트는 dataset_psd 실행용 nohup launcher 다.
# - 공식 dataset 이름은 s-mnist, dvsgesture, shd, deap, forda 다.
# - DATASET 은 단일 값만 허용한다.
# - train/test 전체 입력 PSD 기준선과, 필요하면 deterministic probe_set_reference/ 도 함께 저장한다.
# -----------------------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
TIMESTAMP="${TIMESTAMP:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-}"
CUSTOM_TIMESTAMP="${CUSTOM_TIMESTAMP:-}"

PSD_PLOT_WRITER_WORKERS="${PSD_PLOT_WRITER_WORKERS:-1}"
PSD_PLOT_QUEUE_MAXSIZE="${PSD_PLOT_QUEUE_MAXSIZE:-8}"
PSD_PLOT_WRITER_DPI="${PSD_PLOT_WRITER_DPI:-180}"
PSD_PLOT_SKIP_EXISTING="${PSD_PLOT_SKIP_EXISTING:-0}"
PSD_PLOT_WRITER_START_METHOD="${PSD_PLOT_WRITER_START_METHOD:-spawn}"

GPU=""
SEED=""
BATCH_SIZE=""
NUM_WORKERS=""
DOWNLOAD=""
DATASET=""
SHD_T=""
SHD_MAX_TIME=""
SHD_BINNING=""
SHD_UNIT_INDEXING=""
SHD_CHANNEL_FLIP=""
SHD_ALIGN_TO_FIRST_EVENT=""
SHD_USE_EVENT_COUNTS=""
DVSGESTURE_CHUNK_SIZE=""
DVSGESTURE_EMPTY_SIZE=""
DVSGESTURE_DT_MS=""
DVSGESTURE_DS=""
DEAP_LABEL_AXIS=""
DEAP_NUM_CLASSES=""
PSD_WINDOW=""
PSD_OVERLAP=""
USERBIN_EDGES=()
SAME_LABEL_N_PER_LABEL=""
BALANCED_GLOBAL_N_PER_LABEL=""
PROBE_PLOT=""
MAX_SAMPLES=""

# Append one scalar CLI option only when the value is non-empty.
#
# This keeps optional dataset_psd flags out of the final command when the user
# intentionally left them blank in the wrapper configuration.
append_scalar_if_nonempty() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    CMD+=("${flag}" "${value}")
  fi
}

# Append one multi-value CLI option only when the array has elements.
#
# Bash arrays map directly to argparse ``nargs`` parameters, so this helper keeps
# the command assembly readable and avoids repeated length checks.
append_array_if_nonempty() {
  local flag="$1"
  local -n values_ref="$2"
  if [[ ${#values_ref[@]} -gt 0 ]]; then
    CMD+=("${flag}" "${values_ref[@]}")
  fi
}

# Read only the launcher-owned CLI overrides from the user command line.
#
# dataset_psd.sh owns root-path settings such as ``--data_root`` and
# ``--out_root``. Everything else is forwarded to the internal Python entry
# unchanged.
apply_wrapper_overrides_from_cli() {
  local -a args=("$@")
  local i=0
  while [[ ${i} -lt ${#args[@]} ]]; do
    case "${args[i]}" in
      --data_root)
        ((i + 1 < ${#args[@]})) || { echo "--data_root requires a value" >&2; exit 1; }
        DATA_ROOT="${args[i + 1]}"
        ((i += 2))
        ;;
      --data_root=*)
        DATA_ROOT="${args[i]#*=}"
        ((i += 1))
        ;;
      --out_root)
        ((i + 1 < ${#args[@]})) || { echo "--out_root requires a value" >&2; exit 1; }
        OUT_ROOT="${args[i + 1]}"
        ((i += 2))
        ;;
      --out_root=*)
        OUT_ROOT="${args[i]#*=}"
        ((i += 1))
        ;;
      *)
        ((i += 1))
        ;;
    esac
  done
}

# Fail early when a required wrapper path is not absolute.
#
# The project spec requires external data and output roots to be absolute so the
# generated nohup logs always point to unambiguous filesystem locations.
ensure_absolute_path() {
  local path="$1"
  local name="$2"
  [[ "${path}" = /* ]] || { echo "${name} must be an absolute path" >&2; exit 1; }
}

# Start the internal execution branch with nohup and save its PID.
#
# The wrapper launches itself once more with an internal flag so the parent
# process can stay small and only manage logging / bookkeeping.
launch_background() {
  local log_path="$1"
  local pid_path="$2"
  shift 2

  ROOT_DIR="${ROOT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PSD_PLOT_WRITER_WORKERS="${PSD_PLOT_WRITER_WORKERS}" \
  PSD_PLOT_QUEUE_MAXSIZE="${PSD_PLOT_QUEUE_MAXSIZE}" \
  PSD_PLOT_WRITER_DPI="${PSD_PLOT_WRITER_DPI}" \
  PSD_PLOT_SKIP_EXISTING="${PSD_PLOT_SKIP_EXISTING}" \
  PSD_PLOT_WRITER_START_METHOD="${PSD_PLOT_WRITER_START_METHOD}" \
  nohup bash "$0" "${_INTERNAL_FLAG}" "$@" >"${log_path}" 2>&1 &

  echo $! >"${pid_path}"
}

if [[ "${1:-}" == "${_INTERNAL_FLAG}" ]]; then
  shift
  CMD=("${PYTHON_BIN}" -u "${ROOT_DIR}/src/dataset_psd.py")
  append_scalar_if_nonempty --dataset "${DATASET}"
  append_scalar_if_nonempty --data_root "${DATA_ROOT}"
  append_scalar_if_nonempty --out_root "${OUT_ROOT}"
  append_scalar_if_nonempty --gpu "${GPU}"
  append_scalar_if_nonempty --seed "${SEED}"
  append_scalar_if_nonempty --batch_size "${BATCH_SIZE}"
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
  append_scalar_if_nonempty --psd_window "${PSD_WINDOW}"
  append_scalar_if_nonempty --psd_overlap "${PSD_OVERLAP}"
  append_array_if_nonempty --userbin_edges USERBIN_EDGES
  append_scalar_if_nonempty --same_label_n_per_label "${SAME_LABEL_N_PER_LABEL}"
  append_scalar_if_nonempty --balanced_global_n_per_label "${BALANCED_GLOBAL_N_PER_LABEL}"
  append_scalar_if_nonempty --probe_plot "${PROBE_PLOT}"
  append_scalar_if_nonempty --max_samples "${MAX_SAMPLES}"
  append_scalar_if_nonempty --exp_name "${EXP_NAME}"
  append_scalar_if_nonempty --timestamp "${CUSTOM_TIMESTAMP}"
  CMD+=("$@")
  exec "${CMD[@]}"
fi

apply_wrapper_overrides_from_cli "$@"
[[ -n "${DATA_ROOT}" ]] || { echo "DATA_ROOT is empty. Edit bash/dataset_psd.sh or pass --data_root /abs/path" >&2; exit 1; }
[[ -n "${OUT_ROOT}" ]] || { echo "OUT_ROOT is empty. Edit bash/dataset_psd.sh or pass --out_root /abs/path" >&2; exit 1; }
ensure_absolute_path "${DATA_ROOT}" "DATA_ROOT"
ensure_absolute_path "${OUT_ROOT}" "OUT_ROOT"

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${OUT_ROOT}/log"
fi
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/dataset_psd_${TIMESTAMP}.log"
PID_PATH="${LOG_DIR}/dataset_psd_${TIMESTAMP}.pid"

launch_background "${LOG_PATH}" "${PID_PATH}" "$@"

echo "Started dataset_psd"
echo "PID: $(cat "${PID_PATH}")"
echo "LOG: ${LOG_PATH}"
