#!/usr/bin/env bash

DATASET_DIR="${DATASET_DIR:-dataset}"
WORKERS=16
MAP_NAME="Austin"
EGO_RACELINE="raceline1"
NUM_STARTPOINTS=80
SIM_DURATION=8.0
RENDER=false
OPPONENT_RACELINES=(raceline0 raceline1 raceline2)
OPPONENT_SPEED_SCALES=(0.4 0.6 0.8)
INTERVAL_INDEX=15
MAX_COLLISION_RATE_PERCENT=20
MIN_SCENARIOS_FOR_COLLISION_GUARD=25

shopt -s nullglob
existing_artifacts=(
    "$DATASET_DIR"/success/*.csv
    "$DATASET_DIR"/collision/*.json
)
if (( ${#existing_artifacts[@]} > 0 )); then
    echo "Refusing to mix this collection with existing artifacts in $DATASET_DIR" >&2
    exit 1
fi

mapfile -t scenarios < <(
    python - "$MAP_NAME" "$EGO_RACELINE" "$NUM_STARTPOINTS" \
        "${OPPONENT_RACELINES[*]}" "${OPPONENT_SPEED_SCALES[*]}" <<'PY'
import sys

from utils import collection_scenarios

for scenario in collection_scenarios(
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(), sys.argv[5].split()
):
    print(*scenario)
PY
)
if (( ${#scenarios[@]} == 0 )); then
    echo "No scenarios planned for ${MAP_NAME}/${EGO_RACELINE}" >&2
    exit 1
fi

total_jobs=${#scenarios[@]}
echo "Collecting ${total_jobs} scenarios on ${MAP_NAME} with ${WORKERS} workers"
echo "Stopping if the collision rate exceeds ${MAX_COLLISION_RATE_PERCENT}% after ${MIN_SCENARIOS_FOR_COLLISION_GUARD} completed scenarios"

pids=()
stopped_early=0
last_reported_completed=0

monitor_collision_rate() {
    local collision_count completed_count
    local -a live_collision_files live_success_files
    live_collision_files=("$DATASET_DIR"/collision/*.json)
    live_success_files=("$DATASET_DIR"/success/*.csv)
    collision_count=${#live_collision_files[@]}
    completed_count=$((collision_count + ${#live_success_files[@]}))

    if (( completed_count > last_reported_completed )) \
        && (( completed_count >= last_reported_completed + WORKERS || completed_count == total_jobs )); then
        printf 'Progress: %d/%d complete, %d collisions (%.1f%%)\n' \
            "$completed_count" "$total_jobs" "$collision_count" \
            "$(awk -v collisions="$collision_count" -v completed="$completed_count" 'BEGIN { print 100 * collisions / completed }')"
        last_reported_completed=$completed_count
    fi

    if (( completed_count >= MIN_SCENARIOS_FOR_COLLISION_GUARD )) \
        && (( collision_count * 100 > MAX_COLLISION_RATE_PERCENT * completed_count )); then
        printf 'Stopping early: %d/%d completed, %d collisions (over %d%% threshold)\n' \
            "$completed_count" "$total_jobs" "$collision_count" "$MAX_COLLISION_RATE_PERCENT" >&2
        for pid in "${pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null
        done
        return 1
    fi
}

for scenario in "${scenarios[@]}"; do
    read -r opponent_raceline opponent_speed_scale ego_idx <<< "$scenario"
    while (( $(jobs -rp | wc -l) >= WORKERS )); do
        if ! monitor_collision_rate; then
            stopped_early=1
            break 2
        fi
        sleep 0.1
    done
    python expert.py \
        --map_name "$MAP_NAME" --dataset_dir "$DATASET_DIR" \
        --ego_idx "$ego_idx" --interval_idx "$INTERVAL_INDEX" \
        --opponent_raceline "$opponent_raceline" \
        --opponent_speed_scale "$opponent_speed_scale" \
        --sim_duration "$SIM_DURATION" $([[ "$RENDER" == true ]] && echo --render) \
        >/dev/null &
    pids+=("$!")
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

python - "$DATASET_DIR" "$failures" <<PY || exit 1
import sys
from utils import write_collection_summary

write_collection_summary(
    sys.argv[1],
    {
        "map_name": "$MAP_NAME",
        "ego_raceline": "$EGO_RACELINE",
        "num_startpoints": $NUM_STARTPOINTS,
        "opponent_racelines": "${OPPONENT_RACELINES[*]}".split(),
        "opponent_speed_scales": [float(value) for value in "${OPPONENT_SPEED_SCALES[*]}".split()],
        "interval_index": $INTERVAL_INDEX,
        "simulation_duration_seconds": $SIM_DURATION,
        "render": "$RENDER" == "true",
        "workers": $WORKERS,
    },
    int(sys.argv[2]),
)
PY

if (( stopped_early )); then
    exit 2
fi
if (( failures > 0 )); then
    exit 1
fi
