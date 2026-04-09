#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PSD_CONFIG_SCRIPT="${SCRIPT_DIR}/psd.sh"

# -----------------------------------------------------------------------------
# 실행 개요
# -----------------------------------------------------------------------------
# - 이 스크립트는 SSH 원격 환경용 nohup + background launcher 다.
# - 한 번 실행하면 scenario index 별로 psd.sh 를 병렬 실행한다.
# - scenario parent 는 <OUT_ROOT>/<EXP_NAME>/<scenario_index>/ 구조를 쓴다.
# - 각 scenario 안에서 psd.sh 는 단일 DATASET 위에서 model × readout 조합을 직렬 실행한다.
# - grouped model/readout 문법을 쓰면 각 group 쌍이 하나의 병렬 scenario 로 확장된다.
#   예) SCENARIO_MODEL_GROUPS_RAW="((a b) (c d))"
#       SCENARIO_READOUT_GROUPS_RAW="((1 2) (3 4))"
#   이면 scenario 0 에서 (a b) × (1 2), scenario 1 에서 (c d) × (3 4) 가 각각 직렬 실행된다.
# -----------------------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-}"
EXP_NAME="${EXP_NAME:-}"
LOG_DIR_NAME="${LOG_DIR_NAME:-log}"
TIMESTAMP="${TIMESTAMP:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"

PSD_PLOT_WRITER_WORKERS="${PSD_PLOT_WRITER_WORKERS:-4}"
PSD_PLOT_QUEUE_MAXSIZE="${PSD_PLOT_QUEUE_MAXSIZE:-1024}"
PSD_PLOT_WRITER_DPI="${PSD_PLOT_WRITER_DPI:-120}"
PSD_PLOT_SKIP_EXISTING="${PSD_PLOT_SKIP_EXISTING:-0}"
PSD_PLOT_WRITER_START_METHOD="${PSD_PLOT_WRITER_START_METHOD:-fork}"

# -----------------------------------------------------------------------------
# 병렬 scenario 정의
# -----------------------------------------------------------------------------
# - SCENARIO_COUNT 는 최종 병렬 scenario 수다.
# - SCENARIO_DATASET_TOKENS 의 각 항목은 단일 dataset token 이어야 한다.
# - SCENARIO_MODELS / SCENARIO_READOUT_MODES 는 공백 분리 문자열로 주고,
#   해당 scenario 안에서 model × readout 직렬 조합으로 실행된다.
# - 아래 배열은 길이 0, 1, 또는 SCENARIO_COUNT 를 허용한다.
#   * 길이 0 : psd.sh base 값 사용
#   * 길이 1 : 모든 scenario 로 broadcast
#   * 길이 SCENARIO_COUNT : scenario 별 개별 override
#
# grouped model/readout 예시
#   SCENARIO_COUNT=2
#   SCENARIO_MODEL_GROUPS_RAW="((lif lif_R) (rf rf_R))"
#   SCENARIO_READOUT_GROUPS_RAW="((final_membrane first_spike) (final_membrane))"
#   SCENARIO_DATASET_TOKENS=("s-mnist")
#   SCENARIO_GPUS=(0 1)
#   SCENARIO_HIDDENS=("60 48 36")
# -----------------------------------------------------------------------------
SCENARIO_COUNT=0
SCENARIO_GPUS=()
SCENARIO_DATASET_TOKENS=()
SCENARIO_MODELS=()
SCENARIO_READOUT_MODES=()
SCENARIO_HIDDENS=()
SCENARIO_EXTRA_ARGS=()
SCENARIO_MODEL_GROUPS_RAW="${SCENARIO_MODEL_GROUPS_RAW:-}"
SCENARIO_READOUT_GROUPS_RAW="${SCENARIO_READOUT_GROUPS_RAW:-}"

# Parse wrapper-level CLI overrides that belong to the launcher itself.
#
# This function extracts only run_psd.sh ownership arguments such as data/out
# roots and experiment name. Any remaining arguments are forwarded untouched to
# the underlying psd.sh config script.
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
      --exp_name)
        ((i + 1 < ${#args[@]})) || { echo "--exp_name requires a value" >&2; exit 1; }
        EXP_NAME="${args[i + 1]}"
        ((i += 2))
        ;;
      --exp_name=*)
        EXP_NAME="${args[i]#*=}"
        ((i += 1))
        ;;
      *)
        ((i += 1))
        ;;
    esac
  done
}

# Fail early when a path-valued launcher setting is not absolute.
ensure_absolute_path() {
  local path="$1"
  local name="$2"
  [[ "${path}" = /* ]] || { echo "${name} must be an absolute path" >&2; exit 1; }
}

# Convert one whitespace-delimited shell string into a bash array.
#
# Scenario overrides are stored as plain strings in environment variables, so we
# centralize the split step here to keep later append helpers simple.
split_words_into_array() {
  local text="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ -n "${text}" ]]; then
    # shellcheck disable=SC2206
    out_ref=(${text})
  fi
}

# Append a multi-value CLI flag only when its raw text is non-empty.
append_optional_multi_value_flag() {
  local -n cmd_ref="$1"
  local flag="$2"
  local raw="$3"
  local -a parsed=()
  split_words_into_array "${raw}" parsed
  if [[ ${#parsed[@]} -gt 0 ]]; then
    cmd_ref+=("${flag}" "${parsed[@]}")
  fi
}

# Append a scalar CLI flag only when a concrete value is present.
append_optional_scalar_flag() {
  local -n cmd_ref="$1"
  local flag="$2"
  local raw="$3"
  if [[ -n "${raw}" ]]; then
    cmd_ref+=("${flag}" "${raw}")
  fi
}

# Broadcast one scalar value into an array of fixed scenario length.
fill_array_with_repeated_value() {
  local -n ref="$1"
  local count="$2"
  local value="$3"
  ref=()
  local i
  for ((i=0; i<count; i++)); do
    ref+=("${value}")
  done
}

# Normalize scenario override arrays to exactly SCENARIO_COUNT entries.
#
# Accepted forms are: empty (optional broadcast fill), one value (broadcast to
# all scenarios), or one value per scenario. Any other length is an error.
normalize_scenario_array() {
  local name="$1"
  local expected="$2"
  local allow_empty="$3"
  local empty_fill="$4"
  local -n ref="$name"

  if [[ ${#ref[@]} -eq ${expected} ]]; then
    return
  fi
  if [[ ${#ref[@]} -eq 0 ]]; then
    if [[ "${allow_empty}" == "1" ]]; then
      fill_array_with_repeated_value "$name" "${expected}" "${empty_fill}"
      return
    fi
    echo "${name} must not be empty" >&2
    exit 1
  fi
  if [[ ${#ref[@]} -eq 1 ]]; then
    fill_array_with_repeated_value "$name" "${expected}" "${ref[0]}"
    return
  fi
  echo "${name} length must be 0, 1, or ${expected}; got ${#ref[@]}" >&2
  exit 1
}

# Parse the grouped "((a b) (c d))" scenario syntax into one bash array entry
# per group. The small Python helper keeps the parenthesis parser reliable and
# much easier to maintain than pure bash string surgery.
parse_grouped_token_lists() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ -z "${raw}" ]]; then
    return
  fi
  mapfile -t out_ref < <("${PYTHON_BIN}" - "${raw}" <<'PY'
import re
import sys

raw = sys.argv[1].strip()
if not raw:
    raise SystemExit(0)

tokens = re.findall(r'\(|\)|[^\s()]+', raw)
pos = 0

def parse_node():
    global pos
    if pos >= len(tokens) or tokens[pos] != '(':
        raise SystemExit('group parser expected "("')
    pos += 1
    items = []
    while pos < len(tokens) and tokens[pos] != ')':
        tok = tokens[pos]
        if tok == '(':
            items.append(parse_node())
        else:
            items.append(tok)
            pos += 1
    if pos >= len(tokens) or tokens[pos] != ')':
        raise SystemExit('group parser found unbalanced parentheses')
    pos += 1
    return items

node = parse_node()
if pos != len(tokens):
    raise SystemExit('group parser found trailing tokens')
if any(not isinstance(group, list) for group in node):
    node = [node]
for group in node:
    if any(isinstance(x, list) for x in group):
        raise SystemExit('group parser supports exactly two nesting levels')
    print(' '.join(str(x) for x in group))
PY
  )
}

# Expand grouped model/readout syntax into ordinary per-scenario arrays.
expand_grouped_model_readout_scenarios() {
  if [[ -z "${SCENARIO_MODEL_GROUPS_RAW}" && -z "${SCENARIO_READOUT_GROUPS_RAW}" ]]; then
    return
  fi
  [[ -n "${SCENARIO_MODEL_GROUPS_RAW}" && -n "${SCENARIO_READOUT_GROUPS_RAW}" ]] || {
    echo "Both SCENARIO_MODEL_GROUPS_RAW and SCENARIO_READOUT_GROUPS_RAW must be set together" >&2
    exit 1
  }

  local -a model_groups=()
  local -a readout_groups=()
  parse_grouped_token_lists "${SCENARIO_MODEL_GROUPS_RAW}" model_groups
  parse_grouped_token_lists "${SCENARIO_READOUT_GROUPS_RAW}" readout_groups

  [[ ${#model_groups[@]} -eq ${#readout_groups[@]} ]] || {
    echo "Grouped model/readout scenario counts differ: ${#model_groups[@]} vs ${#readout_groups[@]}" >&2
    exit 1
  }
  [[ ${#model_groups[@]} -gt 0 ]] || {
    echo "Grouped model/readout syntax produced zero scenarios" >&2
    exit 1
  }
  if [[ ${SCENARIO_COUNT} -ne 0 && ${SCENARIO_COUNT} -ne ${#model_groups[@]} ]]; then
    echo "SCENARIO_COUNT=${SCENARIO_COUNT} conflicts with grouped scenario count ${#model_groups[@]}" >&2
    exit 1
  fi

  SCENARIO_COUNT=${#model_groups[@]}
  SCENARIO_MODELS=("${model_groups[@]}")
  SCENARIO_READOUT_MODES=("${readout_groups[@]}")
}

# Launch exactly one scenario under nohup and persist its PID/log paths.
launch_one() {
  local index="$1"
  shift
  local scenario_root="${OUT_ROOT}/${EXP_NAME}/${index}"
  local log_dir="${scenario_root}/${LOG_DIR_NAME}"
  local log_path="${log_dir}/psd_${TIMESTAMP}.log"
  local pid_path="${log_dir}/psd_${TIMESTAMP}.pid"
  local -a launch_cmd=(bash "${PSD_CONFIG_SCRIPT}")
  local -a scenario_extra=()

  append_optional_scalar_flag launch_cmd --data_root "${DATA_ROOT}"
  append_optional_scalar_flag launch_cmd --out_root "${scenario_root}"
  append_optional_scalar_flag launch_cmd --exp_name "${EXP_NAME}"
  append_optional_scalar_flag launch_cmd --gpu "${SCENARIO_GPUS[index]}"
  append_optional_scalar_flag launch_cmd --dataset "${SCENARIO_DATASET_TOKENS[index]}"
  append_optional_multi_value_flag launch_cmd --model "${SCENARIO_MODELS[index]}"
  append_optional_multi_value_flag launch_cmd --readout_mode "${SCENARIO_READOUT_MODES[index]}"
  append_optional_multi_value_flag launch_cmd --hidden "${SCENARIO_HIDDENS[index]}"
  split_words_into_array "${SCENARIO_EXTRA_ARGS[index]}" scenario_extra
  if [[ ${#scenario_extra[@]} -gt 0 ]]; then
    launch_cmd+=("${scenario_extra[@]}")
  fi

  launch_cmd+=("$@")
  mkdir -p "${scenario_root}" "${log_dir}"

  ROOT_DIR="${ROOT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PSD_RUN_LAUNCHER_ACTIVE=1 \
  PSD_PLOT_WRITER_WORKERS="${PSD_PLOT_WRITER_WORKERS}" \
  PSD_PLOT_QUEUE_MAXSIZE="${PSD_PLOT_QUEUE_MAXSIZE}" \
  PSD_PLOT_WRITER_DPI="${PSD_PLOT_WRITER_DPI}" \
  PSD_PLOT_SKIP_EXISTING="${PSD_PLOT_SKIP_EXISTING}" \
  PSD_PLOT_WRITER_START_METHOD="${PSD_PLOT_WRITER_START_METHOD}" \
  nohup "${launch_cmd[@]}" >"${log_path}" 2>&1 &

  echo $! >"${pid_path}"

  echo "Started psd scenario ${index}"
  echo "PID: $(cat "${pid_path}")"
  echo "LOG: ${log_path}"
  echo "OUT: ${scenario_root}"
  echo "----------------------------------------"
}

apply_wrapper_overrides_from_cli "$@"
[[ -f "${PSD_CONFIG_SCRIPT}" ]] || { echo "Missing config script: ${PSD_CONFIG_SCRIPT}" >&2; exit 1; }
expand_grouped_model_readout_scenarios
[[ ${SCENARIO_COUNT} -gt 0 ]] || { echo "SCENARIO_COUNT must be >= 1" >&2; exit 1; }

normalize_scenario_array SCENARIO_GPUS "${SCENARIO_COUNT}" 1 ""
normalize_scenario_array SCENARIO_DATASET_TOKENS "${SCENARIO_COUNT}" 1 ""
normalize_scenario_array SCENARIO_MODELS "${SCENARIO_COUNT}" 1 ""
normalize_scenario_array SCENARIO_READOUT_MODES "${SCENARIO_COUNT}" 1 ""
normalize_scenario_array SCENARIO_HIDDENS "${SCENARIO_COUNT}" 1 ""
normalize_scenario_array SCENARIO_EXTRA_ARGS "${SCENARIO_COUNT}" 1 ""

[[ -n "${OUT_ROOT}" ]] || { echo "OUT_ROOT is empty. Edit bash/run_psd.sh or pass --out_root /abs/path" >&2; exit 1; }
[[ -n "${EXP_NAME}" ]] || { echo "EXP_NAME is empty. Edit bash/run_psd.sh or pass --exp_name <experiment_name>" >&2; exit 1; }
if [[ -n "${DATA_ROOT}" ]]; then
  ensure_absolute_path "${DATA_ROOT}" "DATA_ROOT"
fi
ensure_absolute_path "${OUT_ROOT}" "OUT_ROOT"
mkdir -p "${OUT_ROOT}"

for ((i=0; i<SCENARIO_COUNT; i++)); do
  launch_one "${i}" "$@"
done
