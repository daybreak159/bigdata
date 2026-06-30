#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../源码/recommendation_project"

cd "$PROJECT_DIR"
python3 src/predict_final.py --model optimized_ensemble
cp results/final_result.txt "$SCRIPT_DIR/../实验结果/final_result.txt"

echo "Generated: $SCRIPT_DIR/../实验结果/final_result.txt"
