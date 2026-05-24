#!/usr/bin/env bash
# 단계 실행용 JSON 래퍼 스크립트
set -euo pipefail
CONFIG_PATH="${1:-config/model_training.json}"
python src/model_training.py --config "$CONFIG_PATH"
