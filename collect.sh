#!/usr/bin/env bash

set -u -o pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "end2race" ]]; then
    echo "Activate the Python 3.11 end2race environment first: conda activate end2race" >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "end2race requires Python 3.11"' || exit 1

WORKERS="${WORKERS:-12}"
MAP_NAME="Austin"
DATASET_DIR="${DATASET_DIR:-dataset}"
EGO_RACELINE="raceline1"
NUM_STARTPOINTS=80
SIM_DURATION=8.0
RENDER=false
OPPONENT_RACELINES=(raceline0 raceline1 raceline2)
OPPONENT_SPEED_SCALES=(0.4 0.6 0.8)
INTERVAL_INDEX=15
MAX_COLLISION_RATE_PERCENT="${MAX_COLLISION_RATE_PERCENT:-20}"
MIN_SCENARIOS_FOR_COLLISION_GUARD="${MIN_SCENARIOS_FOR_COLLISION_GUARD:-25}"

if ! [[ "$MAX_COLLISION_RATE_PERCENT" =~ ^[0-9]+$ ]] || ! [[ "$MIN_SCENARIOS_FOR_COLLISION_GUARD" =~ ^[0-9]+$ ]]; then
    echo "Collision-rate guard settings must be nonnegative integers" >&2
    exit 1
fi

shopt -s nullglob
existing_artifacts=(
    "$DATASET_DIR"/success/*.csv
    "$DATASET_DIR"/collision/*.json
)
if (( ${#existing_artifacts[@]} > 0 )); then
    echo "Refusing to mix this collection with existing artifacts in $DATASET_DIR" >&2
    exit 1
fi

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

total_jobs=$((${#ego_indices[@]} * ${#OPPONENT_RACELINES[@]} * ${#OPPONENT_SPEED_SCALES[@]}))
echo "Collecting ${total_jobs} scenarios on ${MAP_NAME} with ${WORKERS} workers"
echo "Stopping if the collision rate exceeds ${MAX_COLLISION_RATE_PERCENT}% after ${MIN_SCENARIOS_FOR_COLLISION_GUARD} completed scenarios"

pids=()
stopped_early=0
last_reported_completed=0

stop_active_workers() {
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid"
        fi
    done
}

monitor_collision_rate() {
    local collision_count success_count completed_count
    local -a live_collision_files live_success_files
    live_collision_files=("$DATASET_DIR"/collision/*.json)
    live_success_files=("$DATASET_DIR"/success/*.csv)
    collision_count=${#live_collision_files[@]}
    success_count=${#live_success_files[@]}
    completed_count=$((collision_count + success_count))

    if (( completed_count >= last_reported_completed + WORKERS || completed_count == total_jobs )); then
        printf 'Progress: %d/%d complete, %d collisions (%.1f%%)\n' \
            "$completed_count" "$total_jobs" "$collision_count" \
            "$(awk -v collisions="$collision_count" -v completed="$completed_count" 'BEGIN { print 100 * collisions / completed }')"
        last_reported_completed=$completed_count
    fi

    if (( completed_count >= MIN_SCENARIOS_FOR_COLLISION_GUARD )) \
        && (( collision_count * 100 > MAX_COLLISION_RATE_PERCENT * completed_count )); then
        printf 'Stopping early: %d/%d completed, %d collisions (over %d%% threshold)\n' \
            "$completed_count" "$total_jobs" "$collision_count" "$MAX_COLLISION_RATE_PERCENT" >&2
        stop_active_workers
        return 1
    fi
}

for opponent_raceline in "${OPPONENT_RACELINES[@]}"; do
    for opponent_speed_scale in "${OPPONENT_SPEED_SCALES[@]}"; do
        for ego_idx in "${ego_indices[@]}"; do
            while (( $(jobs -rp | wc -l) >= WORKERS )); do
                if ! monitor_collision_rate; then
                    stopped_early=1
                    break 3
                fi
                sleep 0.1
            done
            python collect.py \
                "$MAP_NAME" "$DATASET_DIR" "$ego_idx" "$INTERVAL_INDEX" \
                "$opponent_raceline" "$opponent_speed_scale" \
                "$SIM_DURATION" "$RENDER" \
                >/dev/null &
            pids+=("$!")
        done
    done
done

while (( $(jobs -rp | wc -l) > 0 )); do
    if ! monitor_collision_rate; then
        stopped_early=1
        break
    fi
    sleep 0.1
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        ((failures += 1))
    fi
done

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
if ! python summarize_dataset.py "$DATASET_DIR" \
    --map-name "$MAP_NAME" \
    --ego-raceline "$EGO_RACELINE" \
    --num-startpoints "$NUM_STARTPOINTS" \
    --sim-duration "$SIM_DURATION" \
    --render "$RENDER" \
    --interval-index "$INTERVAL_INDEX" \
    --workers "$WORKERS" \
    --opponent-racelines "${OPPONENT_RACELINES[@]}" \
    --opponent-speed-scales "${OPPONENT_SPEED_SCALES[@]}" \
    --collection-failures "$failures"; then
    echo "Failed to generate ${DATASET_DIR}/summary.json" >&2
    exit 1
fi
if (( stopped_early )); then
    exit 2
fi
if (( failures > 0 )); then
    exit 1
fi
