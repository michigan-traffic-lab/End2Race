#!/usr/bin/env bash

WORKERS=12
MAP_NAME="Austin"
CHECKPOINT_PATH="${1:-}"
OUTPUT_ROOT="${2:-eval_results}"
EGO_RACELINE="raceline1"
NUM_STARTPOINTS=80
SIM_DURATION=8.0
NOISE=0.0
SEED=42
RENDER=false
OPPONENT_RACELINES=(raceline0 raceline1 raceline2)
OPPONENT_SPEED_SCALES=(0.4 0.6 0.8)
MIN_SCENARIOS_BEFORE_COLLISION_GUARD=360

if [[ "$RENDER" == true ]]; then
    RENDER_FLAG="--render"
else
    RENDER_FLAG=""
fi

if [[ -z "$CHECKPOINT_PATH" || ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

model_name=$(basename "$CHECKPOINT_PATH")
model_name="${model_name%.*}"
OUTPUT_DIR="$OUTPUT_ROOT/$model_name/$MAP_NAME"
mkdir -p "$OUTPUT_DIR" || exit 1

mapfile -t ego_indices < <(
    python -c 'import sys; from utils import get_ego_idx_range; print(*get_ego_idx_range(sys.argv[1], sys.argv[2], int(sys.argv[3])), sep="\n")' \
        "$MAP_NAME" "$EGO_RACELINE" "$NUM_STARTPOINTS"
)
if (( ${#ego_indices[@]} == 0 )); then
    echo "No start points found for ${MAP_NAME}/${EGO_RACELINE}" >&2
    exit 1
fi

results_dir=$(mktemp -d /tmp/end2race-eval.XXXXXX) || exit 1

total_jobs=$((${#ego_indices[@]} * ${#OPPONENT_RACELINES[@]} * ${#OPPONENT_SPEED_SCALES[@]}))
stop_reason="completed"
summary_written=0

write_summary() {
    (( summary_written )) && return 0
    summary_written=1
    python -c 'import sys; from utils import write_multi_evaluation_summary; raise SystemExit(not write_multi_evaluation_summary(*sys.argv[1:]))' \
        "$results_dir" "$OUTPUT_DIR/results.json" "$CHECKPOINT_PATH" \
        "$MAP_NAME" "$EGO_RACELINE" "$NUM_STARTPOINTS" \
        "${OPPONENT_RACELINES[*]}" "${OPPONENT_SPEED_SCALES[*]}" \
        "$SIM_DURATION" "$NOISE" "$SEED" "$total_jobs" "$stop_reason"
}

stop_active_workers() {
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null
            kill -TERM "$pid" 2>/dev/null
        fi
    done
}

cleanup() {
    write_summary
    if [[ "$results_dir" == /tmp/end2race-eval.* && -d "$results_dir" && ! -L "$results_dir" ]]; then
        rm -rf -- "${results_dir:?}"
    fi
}
trap cleanup EXIT
trap 'stop_reason="interrupted"; stop_active_workers; exit 130' INT
trap 'stop_reason="interrupted"; stop_active_workers; exit 143' TERM

echo "Evaluating ${total_jobs} scenarios on ${MAP_NAME} with ${WORKERS} workers"
echo "Artifacts: ${OUTPUT_DIR}"

pids=()
job_id=0
stopped_early=0

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
                    break 4
                fi
                sleep 0.1
            done
            (
                python eval_multi.py \
                    --map_name "$MAP_NAME" \
                    --checkpoint_path "$CHECKPOINT_PATH" \
                    --output_dir "$OUTPUT_DIR" \
                    --ego_raceline "$EGO_RACELINE" \
                    --ego_idx "$ego_idx" \
                    --opponent_raceline "$opponent_raceline" \
                    --opponent_speed_scale "$opponent_speed_scale" \
                    --sim_duration "$SIM_DURATION" \
                    --noise "$NOISE" \
                    --seed "$SEED" \
                    $RENDER_FLAG \
                    >"$results_dir/$job_id.out" 2>"$results_dir/$job_id.err"
                echo "$?" >"$results_dir/$job_id.status"
            ) &
            pids+=("$!")
            ((job_id += 1))
        done
    done
done

if (( ! stopped_early )); then
    while (( $(jobs -rp | wc -l) > 0 )); do
        if ! monitor_collision_rate; then
            stopped_early=1
            break
        fi
        sleep 0.1
    done
fi

for pid in "${pids[@]}"; do
    wait "$pid"
done

if (( ! stopped_early )) && ! monitor_collision_rate; then
    stopped_early=1
fi

if (( stopped_early )); then
    stop_reason="collision_guard"
    write_summary
    exit 2
fi

if write_summary; then
    exit 0
fi
exit 1
