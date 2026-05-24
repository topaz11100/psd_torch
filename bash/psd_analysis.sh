#!/usr/bin/env bash
# 단계 실행용 JSON 래퍼 스크립트
set -euo pipefail
CONFIG_PATH="${1:-config/psd_analysis.json}"
python src/psd_analysis.py --config "$CONFIG_PATH"
