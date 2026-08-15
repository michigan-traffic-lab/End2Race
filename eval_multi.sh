#!/usr/bin/env bash

set -u -o pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "end2race" ]]; then
    echo "Activate the Python 3.11 end2race environment first: conda activate end2race" >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "end2race requires Python 3.11"' || exit 1

WORKERS=12
MAP_NAME="Austin"
CHECKPOINT_PATH="${1:-}"
EGO_RACELINE="raceline1"
NUM_STARTPOINTS=80
SIM_DURATION=8.0
NOISE=0.0
SEED=42
RENDER=false
OPPONENT_RACELINES=(raceline0 raceline1 raceline2)
OPPONENT_SPEED_SCALES=(0.4 0.6 0.8)
MIN_SCENARIOS_BEFORE_COLLISION_GUARD=100

if [[ -z "$CHECKPOINT_PATH" || ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
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

results_dir=$(mktemp -d /tmp/end2race-eval.XXXXXX) || exit 1
case "$results_dir" in
    /tmp/end2race-eval.*) ;;
    *) echo "Unexpected temporary directory: $results_dir" >&2; exit 1 ;;
esac
cleanup() {
    if [[ "$results_dir" == /tmp/end2race-eval.* && -d "$results_dir" && ! -L "$results_dir" ]]; then
        rm -rf -- "${results_dir:?}"
    fi
}
trap cleanup EXIT

total_jobs=$((${#ego_indices[@]} * ${#OPPONENT_RACELINES[@]} * ${#OPPONENT_SPEED_SCALES[@]}))
echo "Evaluating ${total_jobs} scenarios on ${MAP_NAME} with ${WORKERS} workers"

pids=()
job_id=0
stopped_early=0

stop_active_workers() {
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid"
        fi
    done
}

monitor_collision_rate() {
    local collision_count completed_count
    local -a completed_files
    completed_files=("$results_dir"/*.status)
    completed_count=${#completed_files[@]}
    collision_count=$(awk -F= '/^STATE=3$/{count += 1} END {print count + 0}' "$results_dir"/*.out 2>/dev/null)
    if (( completed_count >= MIN_SCENARIOS_BEFORE_COLLISION_GUARD && collision_count * 100 > 20 * completed_count )); then
        printf 'Stopping early: %d/%d complete, %d collisions (over 20%% threshold)\n' \
            "$completed_count" "$total_jobs" "$collision_count" >&2
        stop_active_workers
        return 1
    fi
}

for ego_idx in "${ego_indices[@]}"; do
    for opponent_raceline in "${OPPONENT_RACELINES[@]}"; do
        for opponent_speed_scale in "${OPPONENT_SPEED_SCALES[@]}"; do
            while (( $(jobs -rp | wc -l) >= WORKERS )); do
                if ! monitor_collision_rate; then
                    stopped_early=1
                    break 3
                fi
                sleep 0.1
            done
            (
                python eval_multi.py \
                    "$MAP_NAME" "$CHECKPOINT_PATH" "$ego_idx" "$opponent_raceline" \
                    "$opponent_speed_scale" "$SIM_DURATION" "$NOISE" \
                    "$SEED" "$RENDER" \
                    >"$results_dir/$job_id.out" 2>"$results_dir/$job_id.err"
                echo "$?" >"$results_dir/$job_id.status"
            ) &
            pids+=("$!")
            ((job_id += 1))
        done
    done
done

if (( stopped_early )); then
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
    exit 2
fi

while (( $(jobs -rp | wc -l) > 0 )); do
    if ! monitor_collision_rate; then
        stopped_early=1
        break
    fi
    sleep 0.1
done

for pid in "${pids[@]}"; do
    wait "$pid" || true
done

if (( stopped_early )); then
    exit 2
fi

following=0
overtaking=0
collisions=0
errors=0
for ((index = 0; index < total_jobs; index++)); do
    status=$(<"$results_dir/$index.status")
    if [[ "$status" != 0 ]]; then
        ((errors += 1))
        cat "$results_dir/$index.err" >&2
        continue
    fi
    state=$(awk -F= '/^STATE=/{print $2}' "$results_dir/$index.out")
    case "$state" in
        1) ((following += 1)) ;;
        2) ((overtaking += 1)) ;;
        3) ((collisions += 1)) ;;
        *) ((errors += 1)) ;;
    esac
done

success=$((following + overtaking))
percentage() {
    awk -v count="$1" -v total="$total_jobs" 'BEGIN {printf "%.1f", 100 * count / total}'
}

echo "Evaluation finished"
echo "following: ${following} ($(percentage "$following")%)"
echo "overtaking: ${overtaking} ($(percentage "$overtaking")%)"
echo "success: ${success} ($(percentage "$success")%)"
echo "collision: ${collisions} ($(percentage "$collisions")%)"
echo "errors: ${errors}"
if (( errors > 0 )); then
    exit 1
fi
