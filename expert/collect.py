import argparse
from contextlib import redirect_stderr
import csv
from io import StringIO
import json
from pathlib import Path

with redirect_stderr(StringIO()):
    import f110_gym  # Registers the F1TENTH Gym environment.
    import gym
import imageio
import numpy as np
from f110_gym.envs.base_classes import Integrator

from expert.lattice_planner import create_expert_planner, create_opponent
from expert.utils import (
    create_planner_render_callback,
    downsample_lidar,
    find_opponent_start_index,
    project_point_to_centerline,
    raceline_pose,
    require_end2race_runtime,
    unwrap_progress,
)
from f1tenth_sim.utils import load_racetrack_config, simulation_config
from imitation.model import End2Race

EGO_RACELINE = "raceline1"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect one multi-agent expert demonstration")

    parser.add_argument("--map_name", default="Austin")
    parser.add_argument("--dataset_dir", default="dataset")

    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--interval_idx", type=int, default=15)
    parser.add_argument("--opponent_raceline", default="raceline1")
    parser.add_argument("--opponent_speed_scale", type=float, default=0.8)

    parser.add_argument("--sim_duration", type=float, default=8.0)
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def total_simulation_steps(sim_duration, timestep):
    step_count = round(sim_duration / timestep)
    if not np.isclose(step_count * timestep, sim_duration, rtol=0.0, atol=1e-9):
        raise SystemExit("sim_duration must be divisible by the simulation timestep")
    return step_count


def save_data(
    args, collected_data, video_frames, collision_occurred,
    final_state, base_filename, elapsed_time, opponent_idx, video_fps,
):
    dataset_dir = Path(args.dataset_dir)
    if collision_occurred:
        collision_dir = dataset_dir / "collision"
        collision_dir.mkdir(parents=True, exist_ok=True)
        collision_metadata = {
            "mode": "multi_agent",
            "ego_raceline": EGO_RACELINE,
            "ego_idx": args.ego_idx,
            "opp_raceline": args.opponent_raceline,
            "opp_idx": int(opponent_idx),
            "speed_scale": args.opponent_speed_scale,
            "interval_idx": args.interval_idx,
            "simulation_time": float(elapsed_time),
            "final_state": final_state,
        }

        metadata_path = collision_dir / f"{base_filename}.json"
        metadata_path.write_text(json.dumps(collision_metadata, indent=2), encoding="utf-8")

        if video_frames:
            video_path = collision_dir / f"{base_filename}.mp4"
            imageio.mimwrite(video_path, video_frames, fps=video_fps, macro_block_size=1)
            print(f"Collision video saved to {video_path}")

        print(f"Collision metadata saved to {metadata_path}")
        return

    success_dir = dataset_dir / "success"
    success_dir.mkdir(parents=True, exist_ok=True)
    csv_path = success_dir / f"{base_filename}.csv"
    header = ["time", "current_speed", "steer", "desired_speed"] + [
        f"lidar_{index}" for index in range(End2Race.NUM_LIDAR_FEATURES)
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(collected_data)

    print(f"Multi-agent data saved to {csv_path}")
    if video_frames:
        video_path = success_dir / f"{base_filename}.mp4"
        imageio.mimwrite(video_path, video_frames, fps=video_fps, macro_block_size=1)
        print(f"Video saved to {video_path}")


def collect_scenario(args):
    vehicle = load_racetrack_config().vehicle
    simulation = simulation_config()
    simulation_steps = total_simulation_steps(args.sim_duration, simulation.timestep)

    ego_planner = create_expert_planner(args.map_name, EGO_RACELINE)
    opponent_planner = create_opponent(args.map_name, args.opponent_raceline)
    planner_steps = ego_planner.conf.tracker_steps
    if planner_steps != simulation.steps_per_expert_plan:
        raise ValueError(
            "expert.tracker_steps must match the number of simulation "
            f"steps per expert plan ({simulation.steps_per_expert_plan})"
        )

    env = gym.make(
        "f110-v0",
        map=ego_planner.map_path,
        map_ext=".png",
        timestep=simulation.timestep,
        num_agents=2,
        integrator=Integrator.RK4,
    )

    if args.render:
        render_info = {"ego_steer": 0.0, "ego_speed": 0.0, "opp_steer": 0.0, "opp_speed": 0.0}
        draw_traj_pts = []
        env.add_render_callback(
            create_planner_render_callback(render_info, ego_planner, draw_traj_pts)
        )

    ego_waypoints_xytheta = np.column_stack(
        (ego_planner.waypoints[:, :2], ego_planner.waypoints[:, 3])
    )
    ego_position = raceline_pose(ego_waypoints_xytheta, args.ego_idx)
    opponent_waypoints_xytheta = np.column_stack(
        (opponent_planner.waypoints[:, :2], opponent_planner.waypoints[:, 3])
    )
    opponent_idx = find_opponent_start_index(
        ego_waypoints_xytheta, opponent_waypoints_xytheta, args.ego_idx, args.interval_idx
    )
    opponent_position = raceline_pose(opponent_waypoints_xytheta, opponent_idx)
    initial_velocities = np.asarray([
        simulation.ego_initial_speed_fraction * vehicle.maximum_speed,
        opponent_planner.waypoints[opponent_idx, 2] * args.opponent_speed_scale,
    ])

    # Progress is measured against the ego raceline for both vehicles
    centerline = ego_planner.waypoints[:, :2]
    centerline_total_length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())

    obs, _, done, _ = env.reset(
        poses=np.vstack([ego_position, opponent_position]), velocities=initial_velocities
    )
    if args.render:
        env.render()

    initial_ego_progress, _ = project_point_to_centerline(
        np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
    )
    initial_opponent_progress, _ = project_point_to_centerline(
        np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
    )
    final_state = (
        "overtaking" if initial_ego_progress > initial_opponent_progress else "following"
    )

    simulation_step = 0
    collision_occurred = False
    collected_data = []
    video_frames = []

    while not done and simulation_step < simulation_steps:
        ego_trajectory = ego_planner.plan(
            obs["poses_x"][0],
            obs["poses_y"][0],
            obs["poses_theta"][0],
            obs["scans"][0],
            obs["linear_vels_x"][0],
        )
        opponent_trajectory = opponent_planner.plan(
            obs["poses_x"][1],
            obs["poses_y"][1],
            obs["poses_theta"][1],
            obs["scans"][1],
            obs["linear_vels_x"][1],
        )

        for _ in range(planner_steps):
            if done or simulation_step >= simulation_steps:
                break

            ego_steer, ego_speed = ego_planner.tracker.plan(
                obs["poses_x"][0],
                obs["poses_y"][0],
                obs["poses_theta"][0],
                obs["linear_vels_x"][0],
                ego_trajectory,
            )
            ego_steer = np.clip(ego_steer, -vehicle.steering_limit, vehicle.steering_limit)
            opponent_steer, opponent_speed = opponent_planner.tracker.plan(
                obs["poses_x"][1],
                obs["poses_y"][1],
                obs["poses_theta"][1],
                obs["linear_vels_x"][1],
                opponent_trajectory,
            )
            opponent_steer = np.clip(
                opponent_steer, -vehicle.steering_limit, vehicle.steering_limit
            )
            opponent_speed *= args.opponent_speed_scale
            action = np.asarray([[ego_steer, ego_speed], [opponent_steer, opponent_speed]])

            if args.render:
                render_info.update({
                    "ego_steer": ego_steer,
                    "ego_speed": ego_speed,
                    "opp_steer": opponent_steer,
                    "opp_speed": opponent_speed,
                })

            if simulation_step % simulation.steps_per_control == 0:
                lidar = downsample_lidar(
                    obs["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES
                )
                collected_data.append(
                    [
                        round(len(collected_data) * simulation.control_timestep, 6),
                        obs["linear_vels_x"][0],
                        ego_steer,
                        ego_speed,
                    ]
                    + lidar.tolist()
                )

            obs, _, done, _ = env.step(action)
            simulation_step += 1

            current_ego_progress = unwrap_progress(
                project_point_to_centerline(
                    np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
                )[0],
                initial_ego_progress,
                centerline_total_length,
            )
            current_opponent_progress = unwrap_progress(
                project_point_to_centerline(
                    np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
                )[0],
                initial_opponent_progress,
                centerline_total_length,
            )
            final_state = (
                "overtaking" if current_ego_progress > current_opponent_progress else "following"
            )

            if obs["collisions"][0]:
                done = True
                collision_occurred = True

            if args.render:
                video_frames.append(env.render(mode="rgb_array"))

    elapsed_time = simulation_step * simulation.timestep
    print("Sim elapsed time:", elapsed_time)

    state_prefix = final_state[0]
    opponent_raceline_number = args.opponent_raceline.removeprefix("raceline")
    base_filename = (
        f"{state_prefix}_ol{opponent_raceline_number}_e{args.ego_idx}"
        f"_i{args.interval_idx}_o{opponent_idx}"
        f"_s{args.opponent_speed_scale}"
    )

    save_data(
        args,
        collected_data,
        video_frames,
        collision_occurred,
        final_state,
        base_filename,
        elapsed_time,
        opponent_idx,
        simulation.video_fps,
    )
    ego_planner.close()


def main():
    require_end2race_runtime()
    collect_scenario(parse_arguments())


if __name__ == "__main__":
    main()
