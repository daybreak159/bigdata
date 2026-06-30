#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../源码/recommendation_project"

cd "$PROJECT_DIR"
python3 src/run_experiment.py
