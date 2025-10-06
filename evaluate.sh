#!/bin/bash

# Parameters (converted from argparse defaults)
MODEL_PATH="pretrained/end2race.pth"
HIDDEN_SCALE=4
NOISE=0.0
NUM_WORKERS=4
MAP_NAME="Austin"
RENDER=true
SIM_DURATION=8.0
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
NUM_STARTPOINTS=50

# Generate ego_idx_range
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * max_waypoints / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

# Calculate total segments
total_segments=$((${#ego_idx_range[@]} * ${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]}))

echo "Starting batch evaluation of $total_segments segments"
echo "Model: $MODEL_PATH"
echo "Map: $MAP_NAME"
echo "Workers: $NUM_WORKERS"
echo "Noise level: $NOISE"

start_time=$(date +%s)

# Temporary directory to store individual results
temp_dir=$(mktemp -d)

# Generate parameter combinations and run evaluations
job_id=0

for ego_idx in "${ego_idx_range[@]}"; do
    for opp_raceline in "${OPP_RACELINES[@]}"; do
        for speed_scale in "${OPP_SPEED_SCALES[@]}"; do
            cmd="python eval_multiagent.py --model_path $MODEL_PATH --map_name $MAP_NAME --ego_idx $ego_idx --interval_idx $INTERVAL_IDX --ego_raceline $EGO_RACELINE --opp_raceline $opp_raceline --opp_speedscale $speed_scale --sim_duration $SIM_DURATION --hidden_scale $HIDDEN_SCALE --noise $NOISE"
            
            if [ "$RENDER" = true ]; then
                cmd="$cmd --render"
            fi
            
            while [ $(jobs -r | wc -l) -ge $NUM_WORKERS ]; do
                sleep 0.1
            done
            
            (eval "$cmd" >/dev/null 2>&1; echo $? > "$temp_dir/$job_id") &
            ((job_id++))
        done
    done
done

wait

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "Evaluation complete in ${elapsed} seconds"

# Count results by exit code
following_count=0
overtaking_count=0
collision_count=0
error_count=0

for result_file in "$temp_dir"/*; do
    if [ -f "$result_file" ]; then
        exit_code=$(cat "$result_file")
        case $exit_code in
            1) ((following_count++)) ;;
            2) ((overtaking_count++)) ;;
            3) ((collision_count++)) ;;
            *) ((error_count++)) ;;
        esac
    fi
done

rm -rf "$temp_dir"

success_count=$((following_count + overtaking_count))
success_rate=$(echo "scale=1; $success_count * 100 / $total_segments" | bc)
collision_rate=$(echo "scale=1; $collision_count * 100 / $total_segments" | bc)

echo ""
echo "Results by category:"
echo "  following: $following_count ($(echo "scale=1; $following_count * 100 / $total_segments" | bc)%)"
echo "  overtaking: $overtaking_count ($(echo "scale=1; $overtaking_count * 100 / $total_segments" | bc)%)"
echo "  success: $success_count ($(echo "scale=1; $success_count * 100 / $total_segments" | bc)%)"
echo "  collision: $collision_count ($(echo "scale=1; $collision_count * 100 / $total_segments" | bc)%)"
echo "  error: $error_count ($(echo "scale=1; $error_count * 100 / $total_segments" | bc)%)"
