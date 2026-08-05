#!/usr/bin/env bash

set -u -o pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "end2race" ]]; then
    echo "Activate the Python 3.11 end2race environment first: conda activate end2race" >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "end2race requires Python 3.11"' || exit 1

WORKERS=4
MAP_NAME="Austin"
DATASET_DIR="Dataset_${MAP_NAME}"
EGO_RACELINE="raceline1"
NUM_STARTPOINTS=50
SIM_DURATION=12.0
SAMPLE_INTERVAL=0.1
SEED=6300
RENDER=true
OPPONENT_RACELINES=(raceline0 raceline1 raceline2)
OPPONENT_SPEED_SCALES=(0.2 0.4 0.6 0.8 1.0)
INTERVAL_INDICES=(15 20 25)

mapfile -t ego_indices < <(
    python - "$MAP_NAME" "$EGO_RACELINE" "$NUM_STARTPOINTS" <<'PY'
import sys

from utils import get_ego_idx_range

for index in get_ego_idx_range(
    sys.argv[1], sys.argv[2], int(sys.argv[3])
):
    print(index)
PY
)
if (( ${#ego_indices[@]} == 0 )); then
    echo "No start points found for ${MAP_NAME}/${EGO_RACELINE}" >&2
    exit 1
fi

total_jobs=$((${#ego_indices[@]} * ${#OPPONENT_RACELINES[@]} * ${#OPPONENT_SPEED_SCALES[@]} * ${#INTERVAL_INDICES[@]}))
echo "Collecting ${total_jobs} scenarios on ${MAP_NAME} with ${WORKERS} workers"

pids=()
for interval_idx in "${INTERVAL_INDICES[@]}"; do
    for opponent_raceline in "${OPPONENT_RACELINES[@]}"; do
        for opponent_speed_scale in "${OPPONENT_SPEED_SCALES[@]}"; do
            for ego_idx in "${ego_indices[@]}"; do
                while (( $(jobs -rp | wc -l) >= WORKERS )); do
                    sleep 0.1
                done
                python collect.py \
                    "$MAP_NAME" "$DATASET_DIR" "$ego_idx" "$interval_idx" \
                    "$opponent_raceline" "$opponent_speed_scale" \
                    "$SIM_DURATION" "$SAMPLE_INTERVAL" "$SEED" \
                    "$RENDER" >/dev/null &
                pids+=("$!")
            done
        done
    done
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        ((failures += 1))
    fi
done

shopt -s nullglob
success_files=("$DATASET_DIR"/success/*.csv)
collision_files=("$DATASET_DIR"/collision/*.json)
following=0
for path in "${success_files[@]}"; do
    if [[ $(basename "$path") == f_* ]]; then
        ((following += 1))
    fi
done
overtaking=$((${#success_files[@]} - following))

echo "Collection finished"
echo "following: ${following}"
echo "overtaking: ${overtaking}"
echo "collisions: ${#collision_files[@]}"
echo "failures: ${failures}"
if (( failures > 0 )); then
    exit 1
fi
