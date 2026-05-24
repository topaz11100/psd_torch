#!/usr/bin/env bash
# 단계 실행용 JSON 래퍼 스크립트
set -euo pipefail
CONFIG_PATH="${1:-config/fft2d_analysis.json}"
python src/2d_fft_analysis.py --config "$CONFIG_PATH"
