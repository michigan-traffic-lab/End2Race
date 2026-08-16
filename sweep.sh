#!/usr/bin/env bash

# Training grid. Every combination is repeated REPEATS times with a different run name.
DATASET_DIR="dataset0815"
BATCH_SIZES=(512 1024)
LEARNING_RATES=(0.001 0.0005)
REPEATS=3
NUM_EPOCHS=5000
SAVE_INTERVAL=500
SINGLE_AGENT_MAPS=(Austin Hockenheim MoscowRaceway Nuerburgring)

CHECKPOINT_ROOT="${1:-checkpoint}"
OUTPUT_ROOT="${2:-eval_results}"

records_dir=$(mktemp -d /tmp/end2race-sweep.XXXXXX) || exit 1
records_file="$records_dir/records"
touch "$records_file"

write_summary() {
    mkdir -p "$OUTPUT_ROOT"
    python -c 'import sys; from utils import write_sweep_summary; write_sweep_summary(*sys.argv[1:])' \
        "$records_file" "$OUTPUT_ROOT/sweep.json" "$DATASET_DIR" "$NUM_EPOCHS" \
        "$SAVE_INTERVAL" "${BATCH_SIZES[*]}" "${LEARNING_RATES[*]}" "$REPEATS"
}

cleanup() {
    write_summary
    if [[ "$records_dir" == /tmp/end2race-sweep.* && -d "$records_dir" && ! -L "$records_dir" ]]; then
        rm -rf -- "${records_dir:?}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


record_run() {
    local run="$1" batch_size="$2" learning_rate="$3" repeat="$4" status="$5" qualified="$6"
    {
        echo "TYPE=run"
        echo "RUN=$run"
        echo "BATCH_SIZE=$batch_size"
        echo "LEARNING_RATE=$learning_rate"
        echo "REPEAT=$repeat"
        echo "STATUS=$status"
        echo "MAPS=${SINGLE_AGENT_MAPS[*]}"
        echo "QUALIFIED=$qualified"
        echo "---"
    } >>"$records_file"
}


# Return 1 when the model fails a map and 2 when the evaluator produces no result.
screen_single_agent() {
    local run="$1" checkpoint_path="$2" checkpoint_name="$3" eval_dir="$4"
    local map_name lap_output evaluation_status passed progress
    for map_name in "${SINGLE_AGENT_MAPS[@]}"; do
        lap_output="$records_dir/lap.out"
        python eval_single.py \
            --map_name "$map_name" \
            --checkpoint_path "$checkpoint_path" \
            --output_dir "$eval_dir" >"$lap_output" 2>"$records_dir/lap.err"
        evaluation_status=$?
        if (( evaluation_status == 130 || evaluation_status == 143 )); then
            return "$evaluation_status"
        fi
        if ! grep -q '^PASSED=' "$lap_output"; then
            cat "$records_dir/lap.err" >&2
            return 2
        fi
        {
            echo "TYPE=lap"
            echo "RUN=$run"
            echo "CHECKPOINT=$checkpoint_name"
            echo "MAP_NAME=$map_name"
            grep -E '^[A-Z_]+=' "$lap_output"
            echo "---"
        } >>"$records_file"
        passed=$(awk -F= '$1 == "PASSED" {print $2}' "$lap_output")
        progress=$(awk -F= '$1 == "LAP_PROGRESS" {print $2}' "$lap_output")
        printf '    %-14s passed=%s progress=%s\n' "$map_name" "$passed" "$progress"
        (( passed )) || return 1
    done
}

echo "Sweep: ${#BATCH_SIZES[@]} batch sizes x ${#LEARNING_RATES[@]} learning rates x ${REPEATS} repeats"

for batch_size in "${BATCH_SIZES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
        for ((repeat = 1; repeat <= REPEATS; repeat++)); do
            run="bs${batch_size}_lr${learning_rate}_r${repeat}"
            run_dir="$CHECKPOINT_ROOT/$run"
            echo
            echo "=== $run ==="
            record_run "$run" "$batch_size" "$learning_rate" "$repeat" started ""

            python train.py \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "$run_dir" \
                --save_interval "$SAVE_INTERVAL" \
                --batch_size "$batch_size" \
                --learning_rate "$learning_rate" \
                --num_epochs "$NUM_EPOCHS" || exit $?

            checkpoint_paths=("$run_dir"/epoch_*.pt)
            if [[ ! -f "${checkpoint_paths[0]}" ]]; then
                echo "Training produced no checkpoints in $run_dir" >&2
                exit 1
            fi

            qualified=""
            for checkpoint_path in "${checkpoint_paths[@]}"; do
                checkpoint_name=$(basename "$checkpoint_path")
                echo "  ${checkpoint_name%.pt}"
                screen_single_agent \
                    "$run" "$checkpoint_path" "$checkpoint_name" \
                    "$OUTPUT_ROOT/$run/${checkpoint_name%.pt}"
                single_status=$?
                case "$single_status" in
                    0) ;;
                    1) continue ;;
                    130|143) exit "$single_status" ;;
                    *) exit "$single_status" ;;
                esac

                bash eval_multi.sh "$checkpoint_path" "$OUTPUT_ROOT/$run"
                multi_status=$?
                case "$multi_status" in
                    0) qualified="$checkpoint_path"; break ;;
                    1) exit 1 ;;
                    2) ;;
                    130|143) exit "$multi_status" ;;
                    *) exit "$multi_status" ;;
                esac
            done

            record_run "$run" "$batch_size" "$learning_rate" "$repeat" completed "$qualified"

            if [[ -n "$qualified" ]]; then
                echo "  qualified: $qualified"
            else
                echo "  no qualifying checkpoint"
            fi
        done
    done
done
