#!/usr/bin/env bash

set -u -o pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "end2race" ]]; then
    echo "Activate the Python 3.11 end2race environment first: conda activate end2race" >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "end2race requires Python 3.11"' || exit 1

MAP_NAME="Austin"
CHECKPOINT_PATH="checkpoint_00100.pt"
NOISE=0.0
SEED=42
RENDER=false
LAP_NUM=1
START_IDX=0
MINIMUM_LAP_TIME=10.0

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

exec python eval_single.py \
    "$MAP_NAME" "$CHECKPOINT_PATH" "$NOISE" "$SEED" "$RENDER" \
    "$LAP_NUM" "$START_IDX" "$MINIMUM_LAP_TIME"
