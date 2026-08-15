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
RENDER_FLAG=()
if [[ "$RENDER" == true ]]; then
    RENDER_FLAG=(--render)
fi
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
            python expert.py \
                --map_name "$MAP_NAME" --dataset_dir "$DATASET_DIR" \
                --ego_idx "$ego_idx" --interval_idx "$INTERVAL_INDEX" \
                --opponent_raceline "$opponent_raceline" \
                --opponent_speed_scale "$opponent_speed_scale" \
                --sim_duration "$SIM_DURATION" "${RENDER_FLAG[@]}" \
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

python - "$DATASET_DIR" "$MAP_NAME" "$EGO_RACELINE" "$NUM_STARTPOINTS" \
    "$SIM_DURATION" "$RENDER" "$INTERVAL_INDEX" "$WORKERS" "$failures" \
    "${OPPONENT_RACELINES[*]}" "${OPPONENT_SPEED_SCALES[*]}" <<'PY' || exit 1
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from latticeplanner.lattice_planner import load_lattice_config
from model import End2Race
from utils import (
    CONTROL_FREQUENCY_HZ,
    EXPERT_PLANNER_FREQUENCY_HZ,
    SIMULATION_FREQUENCY_HZ,
    load_racetrack_config,
)

(
    dataset_dir,
    map_name,
    ego_raceline,
    num_startpoints,
    sim_duration,
    render,
    interval_index,
    workers,
    failures,
    racelines,
    speed_scales,
) = sys.argv[1:]
dataset_dir = Path(dataset_dir)
num_startpoints = int(num_startpoints)
sim_duration = float(sim_duration)
render = render == "true"
interval_index = int(interval_index)
workers = int(workers)
failures = int(failures)
opponent_racelines = racelines.split()
opponent_speed_scales = [float(scale) for scale in speed_scales.split()]

success_dir = dataset_dir / "success"
collision_dir = dataset_dir / "collision"
success_paths = list(success_dir.glob("*.csv"))
collision_paths = list(collision_dir.glob("*.json"))


def episode(path, outcome):
    state, raceline, ego_idx, _, _, speed_scale = path.stem.split("_")
    return {
        "outcome": outcome,
        "final_state": "overtaking" if state == "o" else "following",
        "opponent_raceline": f"raceline{raceline[2:]}",
        "ego_idx": int(ego_idx[1:]),
        "speed_scale": float(speed_scale[1:]),
    }


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def tally(episodes):
    collision_free = [
        item for item in episodes if item["outcome"] == "collision_free"
    ]
    overtakes = sum(
        item["final_state"] == "overtaking" for item in collision_free
    )
    return {
        "recorded_scenarios": len(episodes),
        "collision_free_scenarios": len(collision_free),
        "collision_scenarios": len(episodes) - len(collision_free),
        "successful_overtakes": overtakes,
        "collision_free_following": len(collision_free) - overtakes,
        "collision_free_rate_percent": percentage(
            len(collision_free), len(episodes)
        ),
        "successful_overtake_rate_percent": percentage(
            overtakes, len(episodes)
        ),
    }


episodes = [episode(path, "collision_free") for path in success_paths] + [
    episode(path, "collision") for path in collision_paths
]
results = tally(episodes)
summary = {
    "generated_at_utc": datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z"),
    "collection_config": {
        "mode": "multi_agent",
        "map_name": map_name,
        "ego_raceline": ego_raceline,
        "num_startpoints": num_startpoints,
        "ego_indices": sorted({item["ego_idx"] for item in episodes}),
        "opponent_racelines": opponent_racelines,
        "opponent_speed_scales": opponent_speed_scales,
        "interval_index": interval_index,
        "simulation_duration_seconds": sim_duration,
        "simulation_frequency_hz": SIMULATION_FREQUENCY_HZ,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "expert_planner_frequency_hz": EXPERT_PLANNER_FREQUENCY_HZ,
        "render": render,
        "workers": workers,
    },
    "data_config": {
        "lidar_features": End2Race.NUM_LIDAR_FEATURES,
        "csv_columns": 4 + End2Race.NUM_LIDAR_FEATURES,
        "vehicle": vars(load_racetrack_config().vehicle),
        "expert": vars(load_lattice_config().expert),
    },
    "results": {
        "expected_scenarios": num_startpoints
        * len(opponent_racelines)
        * len(opponent_speed_scales),
        "collection_process_failures": failures,
        **results,
        "collisions_while_overtaking": sum(
            item["final_state"] == "overtaking"
            for item in episodes
            if item["outcome"] == "collision"
        ),
        "training_rows": sum(
            sum(1 for _ in path.open(encoding="utf-8")) - 1
            for path in success_paths
        ),
        "success_videos": len(list(success_dir.glob("*.mp4"))),
        "collision_videos": len(list(collision_dir.glob("*.mp4"))),
        "breakdown": [
            {
                "opponent_raceline": raceline,
                "opponent_speed_scale": speed_scale,
                **tally(
                    [
                        item
                        for item in episodes
                        if item["opponent_raceline"] == raceline
                        and item["speed_scale"] == speed_scale
                    ]
                ),
            }
            for raceline in opponent_racelines
            for speed_scale in opponent_speed_scales
        ],
    },
}

summary_path = dataset_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("Collection finished")
print(f"following: {results['collision_free_following']}")
print(f"overtaking: {results['successful_overtakes']}")
print(f"collisions: {results['collision_scenarios']}")
print(f"failures: {failures}")
print(f"Dataset summary saved to {summary_path}")
PY

if (( stopped_early )); then
    exit 2
fi
if (( failures > 0 )); then
    exit 1
fi
