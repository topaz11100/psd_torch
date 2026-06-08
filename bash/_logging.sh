#!/usr/bin/env bash
# Shared logging helper. All stage wrappers default to /home/yongokhan/workspace/logs.
setup_psd_stage_logging() {
  local script_name="${1:-stage}"
  local config_path="${2:-}"
  local log_root="${LOG_ROOT:-/home/yongokhan/workspace/logs}"
  mkdir -p "$log_root"
  local config_stem=""
  if [[ -n "$config_path" ]]; then
    config_stem="$(basename "$config_path")"
    config_stem="${config_stem%.yaml}"
    config_stem="${config_stem%.yml}"
  fi
  if [[ -z "$config_stem" ]]; then
    config_stem="$script_name"
  fi
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$log_root/${script_name}__${config_stem}__${timestamp}.log"
  export PSD_LOG_FILE="$log_file"
  echo "[logging] script=$script_name config=${config_path:-none} log_file=$log_file" >&2
  exec > >(tee -a "$log_file") 2>&1
}
