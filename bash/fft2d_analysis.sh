#!/usr/bin/env bash
# 단계 실행용 JSON 래퍼 스크립트
set -euo pipefail
CONFIG_PATH="${1:-config/fft2d_analysis.json}"
python src/fft2d_analysis.py --config "$CONFIG_PATH"
