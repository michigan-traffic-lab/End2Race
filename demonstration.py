import os
import gym
import f110_gym
import numpy as np
import csv
import json
import argparse
from datetime import datetime
import imageio
from latticeplanner.lattice_planner import *
from latticeplanner.utils import *
from utils import create_planner_render_callback

render_info = {"ego_steer": 0.0, "ego_speed": 0.0, "opp_steer": 0.0, "opp_speed": 0.0}
draw_grid_pts = []
draw_traj_pts = []
draw_target = []
draw_waypoints = []
ego_planner = None
opp_planner = None

def parse_arguments():
    """Argument parser for multi-agent mode"""
    parser = argparse.ArgumentParser(description='Multi-Agent Planner Runner')
    parser.add_argument('--map_name', type=str, default='Austin')
    parser.add_argument('--sim_duration', type=float, default=8.0)
    parser.add_argument('--ego_idx', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--raceline', type=str, default='raceline1')
    parser.add_argument('--opp_speed_scale', type=float, default=0.8)
    parser.add_argument('--interval_idx', type=int, default=15)
    parser.add_argument('--opp_raceline', type=str, default='raceline1')
    
    return parser.parse_args()

def create_render_callback():
    """Create a render callback function for multi-agent mode"""

    def get_ego_planner():
        return ego_planner

    return create_planner_render_callback(render_info, get_ego_planner, draw_grid_pts, draw_traj_pts, margin=800.0)

def setup_ego_planner(map_name, raceline_file, config_path='latticeplanner/lattice_config.yaml'):
    """
    Setup ego vehicle planner with consistent settings
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get map paths
    map_directory, map_path = get_map_paths(map_name)
    ego_raceline_path = os.path.join(map_directory, f"{raceline_file}.csv")
    
    # Create and configure ego planner
    ego_planner = LatticePlanner(config, map_path, ego_raceline_path)
    
    # Use SAME weights regardless of single/multi-agent
    ego_cost_weights = np.array([
        0.12,   # Follow optimization cost     
        2.0,    # Absolute speed reward
        0.3,    # Curvature speed punishment
        0.5     # Opponent collision cost (will be ignored if no opponents)
    ])
    ego_planner.set_parameters({'cost_weights': ego_cost_weights, 'traj_v_scale': 1.0})
    
    return ego_planner, map_directory

def setup_opp_planner(map_name, raceline_file, config_path='latticeplanner/lattice_config.yaml'):
    """
    Setup opponent vehicle planner with consistent settings
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get map paths
    map_directory, map_path = get_map_paths(map_name)
    opp_raceline_path = os.path.join(map_directory, f"{raceline_file}.csv")
    
    # Create and configure opponent planner
    opp_planner = LatticePlanner(config, map_path, opp_raceline_path)
    
    # Set opponent-specific cost weights (defensive/conservative)
    opp_cost_weights = np.array([
        1.0,    # Follow optimization cost 
        2.0,    # Absolute speed reward
        0.5,    # Curvature speed punishment
        0.0     # Opponent collision cost (opponent doesn't avoid)
    ])
    opp_planner.set_parameters({'cost_weights': opp_cost_weights, 'traj_v_scale': 1.0})
    
    return opp_planner

def save_data(args, collected_data, video_frames, collision_occurred, 
              final_state, base_filename, laptime, opp_idx):
    """Save data with unified format"""
    dir_timestamp = datetime.now().strftime("%m%d")
    dataset_dir = f"Dataset_{args.map_name}_{dir_timestamp}"
    
    if collision_occurred:
        collision_dir = os.path.join(dataset_dir, "collision")
        os.makedirs(collision_dir, exist_ok=True)
        
        # Multi-agent collision metadata
        collision_metadata = {
            'mode': 'multi_agent',
            'ego_raceline': str(args.raceline),
            'ego_idx': int(args.ego_idx),
            'opp_raceline': str(args.opp_raceline),
            'opp_idx': int(opp_idx),
            'speed_scale': float(args.opp_speed_scale),
            'interval_idx': int(args.interval_idx),
            'simulation_time': float(laptime),
            'final_state': str(final_state)
        }
        
        metadata_path = os.path.join(collision_dir, f"{base_filename}.json")
        with open(metadata_path, 'w') as f:
            json.dump(collision_metadata, f, indent=2)
        
        if args.render and video_frames:
            video_filename = os.path.join(collision_dir, f"{base_filename}.mp4")
            imageio.mimwrite(video_filename, video_frames, fps=100, macro_block_size=1)
            print(f"Collision video saved to {video_filename}")
        
        print(f"Collision metadata saved to {metadata_path}")
    else:
        success_dir = os.path.join(dataset_dir, "success")
        os.makedirs(success_dir, exist_ok=True)
        
        csv_filename = os.path.join(success_dir, f"{base_filename}.csv")
        
        # Modified header
        header = ["time", "steer", "desired_speed"] + [f"lidar_{i}" for i in range(360)]
        
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(collected_data)
        
        print(f"Multi-agent data saved to {csv_filename}")
        
        if args.render and video_frames:
            video_filename = os.path.join(success_dir, f"{base_filename}.mp4")
            imageio.mimwrite(video_filename, video_frames, fps=100, macro_block_size=1)
            print(f"Video saved to {video_filename}")

def run_lattice_planner(args):
    """Main execution function for multi-agent mode"""
    global ego_planner, opp_planner
    rng = np.random.default_rng(6300)
    
    # Setup planners
    ego_planner, config_directory = setup_ego_planner(args.map_name, args.raceline)
    opp_planner = setup_opp_planner(args.map_name, args.opp_raceline)
    
    # Common setup
    waypoints = ego_planner.waypoints
    ego_wpt_xyhs = np.vstack((ego_planner.waypoints[:, 0], ego_planner.waypoints[:, 1], ego_planner.waypoints[:, 3], ego_planner.waypoints[:, 4])).T
    s_max = waypoints[-1, 4]
    
    # Setup environment
    env = gym.make("f110-v0", map=ego_planner.map_path, map_ext='.png', timestep=0.01, num_agents=2)
    
    if args.render:
        render_callback = create_render_callback()
        env.add_render_callback(render_callback)
    
    # Position setup
    ego_waypoints_xytheta = np.hstack((ego_planner.waypoints[:, :2], ego_planner.waypoints[:, 3].reshape(-1, 1)))
    ego_pos, _ = random_position(ego_waypoints_xytheta, 1, rng, 0.0, 0.0, args.ego_idx, 0)
    
    # Multi-agent initialization
    opp_waypoints_xytheta = np.hstack((opp_planner.waypoints[:, :2], opp_planner.waypoints[:, 3].reshape(-1, 1)))
    opp_wpt_xyhs = np.vstack((opp_planner.waypoints[:, 0], opp_planner.waypoints[:, 1], opp_planner.waypoints[:, 3], opp_planner.waypoints[:, 4])).T
    
    # Find corresponding opponent position
    ego_waypoint = ego_waypoints_xytheta[args.ego_idx]
    ego_map_idx = find_corresponding_waypoint(ego_waypoint, opp_waypoints_xytheta)
    opp_idx = (ego_map_idx + args.interval_idx) % len(opp_waypoints_xytheta)
    opp_pos, _ = random_position(opp_waypoints_xytheta, 1, rng, 0.0, 0.0, opp_idx, 0)
    random_agent_pos = np.vstack([ego_pos, opp_pos])
    
    # Setup centerline for state tracking
    centerline_path = os.path.join(config_directory, 'raceline1.csv')
    centerline_wp = np.loadtxt(centerline_path, delimiter=';', skiprows=1)
    centerline = np.vstack((centerline_wp[:, 1], centerline_wp[:, 2])).T
    centerline_total_length = 0.0
    for i in range(len(centerline) - 1):
        centerline_total_length += np.linalg.norm(centerline[i+1] - centerline[i])
    
    # Reset environment
    obs, _, done, _ = env.reset(poses=random_agent_pos)
    
    if args.render:
        env.render()
    
    # Initialize state tracking
    initial_ego_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][0], obs['poses_y'][0]]), centerline)
    initial_opp_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][1], obs['poses_y'][1]]), centerline)

    if initial_ego_progress > initial_opp_progress:
        initial_state = "overtaking"
    else:
        initial_state = "following"
    
    current_state = initial_state
    final_state = initial_state
    
    # Main simulation variables
    laptime = 0.0
    sim_duration = args.sim_duration
    last_ego_s = 0.0
    last_opp_s = 0.0
    collected_data = []
    sample_interval = 0.1
    next_record_time = sample_interval
    tracker_steps = ego_planner.conf.tracker_steps
    video_frames = []
    collision_occurred = False
    
    # Main simulation loop
    while not done and laptime < sim_duration:
        # Planning phase
        opp_pose = obsDict2oppoArray(obs, 0)
        ego_best_traj = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], opp_pose, obs['linear_vels_x'][0])
        
        opp_pose = obsDict2oppoArray(obs, 1)
        opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], opp_pose, obs['linear_vels_x'][1])
        
        # Tracking loop
        tracker_count = 0
        while not done and tracker_count < tracker_steps:
            # Compute ego control
            ego_steer, ego_speed = ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], ego_best_traj)
            ego_steer = np.clip(ego_steer, -0.52, 0.52)

            # Compute opponent control
            opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], obs['linear_vels_x'][1], opp_best_traj)
            opp_steer = np.clip(opp_steer, -0.52, 0.52)
            opp_speed = opp_speed * args.opp_speed_scale
            action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
            
            # Update render info
            if args.render:
                render_callback.render_info.update({
                    'ego_steer': ego_steer,
                    'ego_speed': ego_speed,
                    'opp_steer': opp_steer,
                    'opp_speed': opp_speed
                })
            
            # Step environment
            obs, timestep, done, _ = env.step(action)
            
            # State tracking
            current_ego_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][0], obs['poses_y'][0]]), centerline)
            current_opp_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][1], obs['poses_y'][1]]), centerline)
            
            if current_ego_progress < initial_ego_progress - centerline_total_length/2:
                current_ego_progress += centerline_total_length
            if current_opp_progress < initial_opp_progress - centerline_total_length/2:
                current_opp_progress += centerline_total_length
            
            if current_ego_progress > current_opp_progress:
                current_state = "overtaking"
            else:
                current_state = "following"
            
            final_state = current_state
            
            # Check collision
            if np.any(obs['collisions']):
                done = True
                collision_occurred = True
            
            # Update time
            laptime += timestep
            if laptime > sim_duration:
                laptime = sim_duration
            
            # Data collection - only ego information
            while laptime >= next_record_time:
                lidar_ego = np.array(obs['scans'][0]).flatten()
                lidar_ego_downsampled = downsample_lidar(lidar_ego, original_points=1440, target_points=360)
                collected_data.append([round(next_record_time, 4), ego_steer, ego_speed] + lidar_ego_downsampled.tolist())
                next_record_time += sample_interval
    
            tracker_count += 1
            
            if args.render:
                video_frames.append(env.render(mode='rgb_array'))
            
            # Update progress tracking
            ego_i = ego_planner.state_i
            ego_s = ego_wpt_xyhs[ego_i, 3]
            if ego_s < last_ego_s:
                ego_s = (ego_s + s_max)

            opp_i = opp_planner.state_i
            opp_s = opp_wpt_xyhs[opp_i, 3]
            if opp_s < last_opp_s:
                opp_s = (opp_s + opp_planner.waypoints[-1, 4])
            last_ego_s, last_opp_s = ego_s, opp_s
    print('Sim elapsed time:', laptime)
    
    # Generate filename
    state_prefix = "o" if final_state == "overtaking" else "f"
    opp_raceline_num = args.opp_raceline.replace('raceline', '').replace('.csv', '')
    base_filename = f"{state_prefix}_ol{opp_raceline_num}_e{args.ego_idx}_o{opp_idx}_s{args.opp_speed_scale}"
    
    # Save data
    save_data(args, collected_data, video_frames, collision_occurred, final_state, base_filename, laptime, opp_idx)

if __name__ == '__main__':
    args = parse_arguments()
    run_lattice_planner(args)