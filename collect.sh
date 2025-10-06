#!/bin/bash

# Parameters (converted from argparse defaults)
WORKERS=4
RENDER=true
MAP_NAME="Austin"
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
SIM_DURATION=8.0
NUM_STARTPOINTS=50
# Generate ego_idx_range
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * (max_waypoints - 1) / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

# Calculate total jobs
total_jobs=$((${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]} * ${#ego_idx_range[@]}))

echo "Lattice Planner Batch Data Collection"
echo "====================================="
echo "Map: $MAP_NAME"
echo "Ego raceline: ${EGO_RACELINE}"
echo "Opponent racelines: ${OPP_RACELINES[*]}"
echo "Speed scales: ${OPP_SPEED_SCALES[*]}"
echo "Interval: ${INTERVAL_IDX}"
echo "Time per run: ${SIM_DURATION}s"
echo "Starting points: $NUM_STARTPOINTS"
echo "Total jobs: $total_jobs"

# Generate parameter combinations and run simulations

for opp_raceline in "${OPP_RACELINES[@]}"; do
    for opp_speed in "${OPP_SPEED_SCALES[@]}"; do
        for ego_idx in "${ego_idx_range[@]}"; do
            cmd="python demonstration.py --map_name $MAP_NAME --raceline $EGO_RACELINE --opp_raceline $opp_raceline --opp_speed_scale $opp_speed --ego_idx $ego_idx --interval_idx $INTERVAL_IDX --sim_duration $SIM_DURATION"

            if [ "$RENDER" = true ]; then
                cmd="$cmd --render"
            fi
            
            while [ $(jobs -r | wc -l) -ge $WORKERS ]; do
                sleep 0.1
            done
            
            eval "$cmd" >/dev/null 2>&1 &
        done
    done
done


wait

echo ""
echo "All simulations completed"

# Find output directories
output_dirs=($(ls -d Dataset_${MAP_NAME}_* 2>/dev/null))
echo "Output directories created: ${#output_dirs[@]}"

# Print basic statistics for each output directory
for output_dir in "${output_dirs[@]}"; do
    echo ""
    echo "Validating: $output_dir"
    
    success_dir="$output_dir/success"
    collision_dir="$output_dir/collision"
    
    success_count=0
    collision_count=0
    follow_count=0
    overtake_count=0
    
    if [ -d "$success_dir" ]; then
        for csv_file in "$success_dir"/*_ol*_e*_o*_s*.csv; do
            if [ -f "$csv_file" ]; then
                filename=$(basename "$csv_file")
                ((success_count++))
                
                if [[ $filename == f_* ]]; then
                    ((follow_count++))
                else
                    ((overtake_count++))
                fi
            fi
        done
    fi
    
    if [ -d "$collision_dir" ]; then
        collision_count=$(ls "$collision_dir"/*.json 2>/dev/null | wc -l)
    fi
    
    total_simulations=$((success_count + collision_count))
    
    echo "  Total simulations: $total_simulations"
    echo "  Successful: $success_count (Follow: $follow_count, Overtake: $overtake_count)"
    echo "  Collisions: $collision_count"
done
