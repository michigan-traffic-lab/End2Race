#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "end2race" ]]; then
    echo "Activate the end2race Conda environment before installing: conda activate end2race" >&2
    exit 1
fi

python -c 'import sys; assert sys.version_info[:2] == (3, 11), "end2race requires Python 3.11"'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install "${PROJECT_DIR}/f1tenth_gym"
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install quadprog==0.1.13 --no-deps
