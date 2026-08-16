import argparse
from pathlib import Path

import gym
import imageio
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from latticeplanner.lattice_planner import create_opponent
from utils import (
    calculate_metrics,
    create_multiagent_render_callback,
    downsample_lidar,
    find_opponent_start_index,
    load_racetrack_config,
    load_raceline,
    mask_lidar_points,
    project_point_to_centerline,
    racetrack_path,
    require_end2race_runtime,
    simulation_config,
    unwrap_progress,
)
from model import End2Race

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate End2Race in one multi-agent scenario")
    # Model and artifacts
    parser.add_argument("--map_name", default="Austin")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--render", action="store_true")

    # Scenario settings
    parser.add_argument("--ego_raceline", default="raceline1")
    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--opponent_raceline", default="raceline1")
    parser.add_argument("--opponent_speed_scale", type=float, default=0.8)
    # Waypoint gap placing the opponent ahead of the ego at reset
    parser.add_argument("--interval_idx", type=int, default=15)
    parser.add_argument("--sim_duration", type=float, default=8.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if (
        args.ego_idx < 0
        or args.opponent_speed_scale <= 0
        or args.sim_duration <= 0
        or not 0 <= args.noise <= 1
    ):
        parser.error(
            "--ego_idx must be nonnegative, --opponent_speed_scale and "
            "--sim_duration must be positive, and --noise must be between 0 and 1"
        )
    return args


def evaluate_segment(model, device, vehicle, args):
    simulation = simulation_config()
    rng = np.random.default_rng(args.seed)

    ego_waypoints = load_raceline(args.map_name, f"{args.ego_raceline}.csv")
    opp_waypoints = (
        ego_waypoints
        if args.opponent_raceline == args.ego_raceline
        else load_raceline(
            args.map_name,
            f"{args.opponent_raceline}.csv",
        )
    )
    opp_idx = find_opponent_start_index(
        ego_waypoints,
        opp_waypoints,
        args.ego_idx,
        args.interval_idx,
    )

    normalized_ego_idx = args.ego_idx % len(ego_waypoints)
    positions = np.array([
        ego_waypoints[normalized_ego_idx, :3],
        opp_waypoints[opp_idx, :3],
    ])
    initial_speed = simulation.ego_initial_speed_fraction * vehicle.maximum_speed
    initial_velocities = np.array(
        [
            initial_speed,
            opp_waypoints[opp_idx, 3] * args.opponent_speed_scale,
        ]
    )

    env = gym.make(
        "f110-v0",
        map=str(racetrack_path(args.map_name, f"{args.map_name}_map")),
        map_ext=".png",
        num_agents=2,
        timestep=simulation.timestep,
        integrator=Integrator.RK4,
    )
    if args.render:
        render_info = {
            "ego_speed": 0.0,
            "ego_steer": 0.0,
            "opp_speed": 0.0,
            "opp_steer": 0.0,
            "state": "unknown",
        }
        visited_points = [[], []]
        drawn_points = [[], []]
        batch_objects = []
        render_callback = create_multiagent_render_callback(
            render_info, visited_points, drawn_points, batch_objects
        )
        env.add_render_callback(render_callback)

    video_frames = []
    opponent = create_opponent(args.map_name, args.opponent_raceline)
    tracker_steps = opponent.conf.tracker_steps
    hidden_size = model.gru.hidden_size
    hidden_state = torch.zeros((1, 1, hidden_size), device=device)
    previous_speed = initial_speed
    control_step = 0
    ego_steer = 0.0
    ego_speed = initial_speed

    centerline_path = racetrack_path(args.map_name, "raceline1.csv")
    centerline_values = np.loadtxt(centerline_path, delimiter=";", skiprows=1)
    centerline = centerline_values[:, 1:3]
    centerline_total_length = np.linalg.norm(
        np.diff(centerline, axis=0), axis=1
    ).sum()

    obs, _, done, _ = env.reset(
        poses=positions,
        velocities=initial_velocities,
    )

    if args.render:
        env.render()

    initial_ego_progress, _ = project_point_to_centerline(
        np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
    )
    initial_opponent_progress, _ = project_point_to_centerline(
        np.array([obs["poses_x"][1], obs["poses_y"][1]]), centerline
    )
    lap_time = 0.0
    collision_occurred = False
    final_state = (
        "overtaking"
        if initial_ego_progress > initial_opponent_progress
        else "following"
    )
    ego_trajectory = []
    speeds = []
    tracker_count = 0
    opponent_trajectory = None

    while not done and lap_time < args.sim_duration:
        if control_step == 0:
            lidar = mask_lidar_points(
                downsample_lidar(
                    obs["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES
                ),
                args.noise,
                rng,
            )

            with torch.no_grad():
                lidar_tensor = torch.as_tensor(
                    lidar, dtype=torch.float32, device=device
                )[None, None]
                speed_tensor = torch.tensor(
                    [[[previous_speed]]], dtype=torch.float32, device=device
                )
                actions, hidden_state = model(
                    lidar_tensor, speed_tensor, hidden_state
                )
                ego_steer = actions[0, -1, 0].item()
                ego_speed = actions[0, -1, 1].item()

            ego_steer = np.clip(
                ego_steer, -vehicle.steering_limit, vehicle.steering_limit
            )
            previous_speed = obs["linear_vels_x"][0]

        if tracker_count == 0:
            opponent_trajectory = opponent.plan(
                obs["poses_x"][1],
                obs["poses_y"][1],
                obs["poses_theta"][1],
                obs["scans"][1],
                obs["linear_vels_x"][1],
            )

        opponent_steer, opponent_speed = opponent.tracker.plan(
            obs["poses_x"][1],
            obs["poses_y"][1],
            obs["poses_theta"][1],
            obs["linear_vels_x"][1],
            opponent_trajectory,
        )
        opponent_steer = np.clip(
            opponent_steer,
            -vehicle.steering_limit,
            vehicle.steering_limit,
        )
        opponent_speed *= args.opponent_speed_scale

        action = np.array(
            [[ego_steer, ego_speed], [opponent_steer, opponent_speed]]
        )
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        control_step = (control_step + 1) % simulation.steps_per_control

        ego_position = [obs["poses_x"][0], obs["poses_y"][0]]
        opponent_position = [obs["poses_x"][1], obs["poses_y"][1]]
        ego_trajectory.append(ego_position)
        speeds.append(obs["linear_vels_x"][0])

        ego_progress = unwrap_progress(
            project_point_to_centerline(
                np.asarray(ego_position), centerline
            )[0],
            initial_ego_progress,
            centerline_total_length,
        )
        opponent_progress = unwrap_progress(
            project_point_to_centerline(
                np.asarray(opponent_position), centerline
            )[0],
            initial_opponent_progress,
            centerline_total_length,
        )

        final_state = (
            "overtaking" if ego_progress > opponent_progress else "following"
        )

        if args.render:
            render_info.update(
                {
                    "ego_speed": ego_speed,
                    "ego_steer": ego_steer,
                    "opp_speed": opponent_speed,
                    "opp_steer": opponent_steer,
                    "state": final_state,
                }
            )
            visited_points[0].append(ego_position)
            visited_points[1].append(opponent_position)
            video_frames.append(env.render(mode="rgb_array"))

        if obs["collisions"][0]:
            collision_occurred = True
            done = True

        tracker_count = (tracker_count + 1) % tracker_steps

    if args.render and video_frames:
        state_prefix = "c" if collision_occurred else final_state[0]
        opponent_raceline_number = args.opponent_raceline.replace("raceline", "")
        noise_suffix = f"_noise{int(args.noise * 100)}" if args.noise else ""
        args.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.output_dir / (
            f"{state_prefix}_ol{opponent_raceline_number}_e{args.ego_idx}"
            f"_o{opp_idx}_s{args.opponent_speed_scale}{noise_suffix}.mp4"
        )
        imageio.mimwrite(
            video_path, video_frames, fps=simulation.video_fps, macro_block_size=1
        )
        print(f"Video saved to {video_path}")

    if args.render:
        for batch_object in batch_objects:
            batch_object.delete()
        type(env.unwrapped).render_callbacks.clear()

    env.close()
    avg_speed, speed_variance, total_distance = calculate_metrics(
        ego_trajectory, speeds
    )
    state = (
        3
        if collision_occurred
        else {"following": 1, "overtaking": 2}[final_state]
    )

    return {
        "state": state,
        "avg_speed": 0.0 if collision_occurred else float(avg_speed),
        "speed_variance": 0.0 if collision_occurred else float(speed_variance),
        "total_distance": float(total_distance),
    }


def main():
    args = parse_arguments()
    require_end2race_runtime()
    vehicle = load_racetrack_config().vehicle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = End2Race().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    result = evaluate_segment(model, device, vehicle, args)
    print(f"STATE={result['state']}")
    print(f"AVG_SPEED={result['avg_speed']:.3f}")
    print(f"SPEED_VARIANCE={result['speed_variance']:.3f}")
    print(f"TOTAL_DISTANCE={result['total_distance']:.3f}")


if __name__ == "__main__":
    main()
