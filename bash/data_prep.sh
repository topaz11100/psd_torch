#!/usr/bin/env bash
# 단계 실행용 JSON 래퍼 스크립트
set -euo pipefail
CONFIG_PATH="${1:-config/data_prep.json}"
python src/data_prep.py --config "$CONFIG_PATH"
