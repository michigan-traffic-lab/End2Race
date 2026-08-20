#!/usr/bin/env bash

WORKERS=16
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
SUPPORTED_MAPS=(Austin Hockenheim MoscowRaceway Nuerburgring)

if [[ "$RENDER" == true ]]; then
    RENDER_FLAG="--render"
else
    RENDER_FLAG=""
fi

if [[ -z "$CHECKPOINT_PATH" || ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

selected_maps=("${@:3}")
if (( ${#selected_maps[@]} == 0 )); then
    selected_maps=("${SUPPORTED_MAPS[@]}")
fi

declare -A supported_maps=()
declare -A seen_maps=()
for map_name in "${SUPPORTED_MAPS[@]}"; do
    supported_maps[$map_name]=1
done
for map_name in "${selected_maps[@]}"; do
    if [[ -z "${supported_maps[$map_name]:-}" ]]; then
        echo "Unsupported map: $map_name" >&2
        exit 1
    fi
    if [[ -n "${seen_maps[$map_name]:-}" ]]; then
        echo "Duplicate map: $map_name" >&2
        exit 1
    fi
    seen_maps[$map_name]=1
done

evaluate_map() (
    map_name=$1
    model_name=$(basename "$CHECKPOINT_PATH")
    model_name="${model_name%.*}"
    output_dir="$OUTPUT_ROOT/$model_name/$map_name"
    mkdir -p "$output_dir" || exit 1

    mapfile -t ego_indices < <(
        python -c 'import sys; from expert.utils import get_ego_idx_range; print(*get_ego_idx_range(sys.argv[1], sys.argv[2], int(sys.argv[3])), sep="\n")' \
            "$map_name" "$EGO_RACELINE" "$NUM_STARTPOINTS"
    )
    if (( ${#ego_indices[@]} == 0 )); then
        echo "No start points found for ${map_name}/${EGO_RACELINE}" >&2
        exit 1
    fi

    results_dir=$(mktemp -d /tmp/end2race-eval.XXXXXX) || exit 1
    total_jobs=$((${#ego_indices[@]} * ${#OPPONENT_RACELINES[@]} * ${#OPPONENT_SPEED_SCALES[@]}))
    stop_reason="completed"
    summary_written=0
    pids=()
    job_id=0

    write_summary() {
        (( summary_written )) && return 0
        summary_written=1
        python -c 'import sys; from expert.utils import write_multi_evaluation_summary; raise SystemExit(not write_multi_evaluation_summary(*sys.argv[1:]))' \
            "$results_dir" "$output_dir/results.json" "$CHECKPOINT_PATH" \
            "$map_name" "$EGO_RACELINE" "$NUM_STARTPOINTS" \
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

    echo "Evaluating ${total_jobs} scenarios on ${map_name} with ${WORKERS} workers"
    echo "Artifacts: ${output_dir}"

    for ego_idx in "${ego_indices[@]}"; do
        for opponent_raceline in "${OPPONENT_RACELINES[@]}"; do
            for opponent_speed_scale in "${OPPONENT_SPEED_SCALES[@]}"; do
                while (( $(jobs -rp | wc -l) >= WORKERS )); do
                    sleep 0.1
                done
                (
                    python -m evaluation.eval_multi \
                        --map_name "$map_name" \
                        --checkpoint_path "$CHECKPOINT_PATH" \
                        --output_dir "$output_dir" \
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

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    if write_summary; then
        exit 0
    fi
    exit 1
)

for map_name in "${selected_maps[@]}"; do
    evaluate_map "$map_name" || exit $?
done
