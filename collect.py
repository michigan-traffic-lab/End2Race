import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import f110_gym  # Registers the F1TENTH Gym environment.
import gym
import imageio
import numpy as np

from config import load_project_config
from latticeplanner.lattice_planner import create_lattice_planner
from latticeplanner.utils import (
    downsample_lidar,
    find_corresponding_waypoint,
    obsDict2oppoArray,
    project_point_to_centerline,
    random_position,
)
from model import End2Race
from utils import (
    SIMULATION_TIMESTEP,
    VIDEO_FPS,
    create_planner_render_callback,
    require_end2race_runtime,
)

EGO_RACELINE = "raceline1"
INTERVAL_INDEX = 15


@dataclass(frozen=True)
class CollectionScenario:
    map_name: str
    dataset_dir: str
    ego_idx: int
    opponent_raceline: str
    opponent_speed_scale: float
    sim_duration: float
    sample_interval: float
    seed: int
    render: bool


def save_data(
    scenario,
    collected_data,
    video_frames,
    collision_occurred,
    final_state,
    base_filename,
    elapsed_time,
    opponent_idx,
):
    dataset_dir = Path(scenario.dataset_dir)
    if collision_occurred:
        collision_dir = dataset_dir / "collision"
        collision_dir.mkdir(parents=True, exist_ok=True)
        collision_metadata = {
            "mode": "multi_agent",
            "ego_raceline": EGO_RACELINE,
            "ego_idx": scenario.ego_idx,
            "opp_raceline": scenario.opponent_raceline,
            "opp_idx": int(opponent_idx),
            "speed_scale": scenario.opponent_speed_scale,
            "interval_idx": INTERVAL_INDEX,
            "simulation_time": float(elapsed_time),
            "final_state": final_state,
        }

        metadata_path = collision_dir / f"{base_filename}.json"
        metadata_path.write_text(
            json.dumps(collision_metadata, indent=2), encoding="utf-8"
        )

        if scenario.render and video_frames:
            video_path = collision_dir / f"{base_filename}.mp4"
            imageio.mimwrite(
                video_path,
                video_frames,
                fps=VIDEO_FPS,
                macro_block_size=1,
            )
            print(f"Collision video saved to {video_path}")

        print(f"Collision metadata saved to {metadata_path}")
        return

    success_dir = dataset_dir / "success"
    success_dir.mkdir(parents=True, exist_ok=True)
    csv_path = success_dir / f"{base_filename}.csv"
    header = ["time", "steer", "desired_speed"] + [
        f"lidar_{index}" for index in range(End2Race.NUM_LIDAR_FEATURES)
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(collected_data)

    print(f"Multi-agent data saved to {csv_path}")
    if scenario.render and video_frames:
        video_path = success_dir / f"{base_filename}.mp4"
        imageio.mimwrite(
            video_path, video_frames, fps=VIDEO_FPS, macro_block_size=1
        )
        print(f"Video saved to {video_path}")


def collect_scenario(vehicle, scenario):
    rng = np.random.default_rng(scenario.seed)

    ego_planner, config_directory = create_lattice_planner(
        scenario.map_name, EGO_RACELINE, "ego"
    )
    opponent_planner, _ = create_lattice_planner(
        scenario.map_name, scenario.opponent_raceline, "opponent"
    )

    env = gym.make(
        "f110-v0",
        map=ego_planner.map_path,
        map_ext=".png",
        timestep=SIMULATION_TIMESTEP,
        num_agents=2,
    )

    render_info = {
        "ego_steer": 0.0,
        "ego_speed": 0.0,
        "opp_steer": 0.0,
        "opp_speed": 0.0,
    }
    draw_grid_pts = []
    draw_traj_pts = []
    if scenario.render:
        render_callback = create_planner_render_callback(
            render_info, ego_planner, draw_grid_pts, draw_traj_pts
        )
        env.add_render_callback(render_callback)

    ego_waypoints_xytheta = np.column_stack(
        (ego_planner.waypoints[:, :2], ego_planner.waypoints[:, 3])
    )
    ego_position, _ = random_position(
        ego_waypoints_xytheta, 1, rng, 0.0, 0.0, scenario.ego_idx, 0
    )
    opponent_waypoints_xytheta = np.column_stack(
        (
            opponent_planner.waypoints[:, :2],
            opponent_planner.waypoints[:, 3],
        )
    )
    ego_waypoint = ego_waypoints_xytheta[scenario.ego_idx]
    ego_map_idx = find_corresponding_waypoint(
        ego_waypoint, opponent_waypoints_xytheta
    )
    opponent_idx = (ego_map_idx + INTERVAL_INDEX) % len(opponent_waypoints_xytheta)
    opponent_pos, _ = random_position(
        opponent_waypoints_xytheta, 1, rng, 0.0, 0.0, opponent_idx, 0
    )
    agent_positions = np.vstack([ego_position, opponent_pos])

    centerline_values = np.loadtxt(
        Path(config_directory) / f"{EGO_RACELINE}.csv",
        delimiter=";",
        skiprows=1,
    )
    centerline = centerline_values[:, 1:3]
    centerline_total_length = np.linalg.norm(
        np.diff(centerline, axis=0), axis=1
    ).sum()

    obs, _, done, _ = env.reset(poses=agent_positions)

    if scenario.render:
        env.render()

    initial_ego_progress, _ = project_point_to_centerline(
        np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
    )
    initial_opponent_progress, _ = project_point_to_centerline(
        np.array([obs["poses_x"][1], obs["poses_y"][1]]), centerline
    )
    final_state = (
        "overtaking"
        if initial_ego_progress > initial_opponent_progress
        else "following"
    )

    elapsed_time = 0.0
    collected_data = []
    next_record_time = scenario.sample_interval
    tracker_steps = ego_planner.conf.tracker_steps
    video_frames = []
    collision_occurred = False

    while not done and elapsed_time < scenario.sim_duration:
        ego_trajectory = ego_planner.plan(
            obs["poses_x"][0],
            obs["poses_y"][0],
            obs["poses_theta"][0],
            obsDict2oppoArray(obs, 0),
            obs["linear_vels_x"][0],
        )
        opponent_trajectory = opponent_planner.plan(
            obs["poses_x"][1],
            obs["poses_y"][1],
            obs["poses_theta"][1],
            obsDict2oppoArray(obs, 1),
            obs["linear_vels_x"][1],
        )

        for _ in range(tracker_steps):
            if done or elapsed_time >= scenario.sim_duration:
                break

            ego_steer, ego_speed = ego_planner.tracker.plan(
                obs["poses_x"][0],
                obs["poses_y"][0],
                obs["poses_theta"][0],
                obs["linear_vels_x"][0],
                ego_trajectory,
            )
            ego_steer = np.clip(
                ego_steer, -vehicle.steering_limit, vehicle.steering_limit
            )

            opponent_steer, opponent_speed = opponent_planner.tracker.plan(
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
            opponent_speed *= scenario.opponent_speed_scale
            action = np.array(
                [[ego_steer, ego_speed], [opponent_steer, opponent_speed]]
            )

            if scenario.render:
                render_info.update(
                    {
                        "ego_steer": ego_steer,
                        "ego_speed": ego_speed,
                        "opp_steer": opponent_steer,
                        "opp_speed": opponent_speed,
                    }
                )

            obs, timestep, done, _ = env.step(action)

            current_ego_progress, _ = project_point_to_centerline(
                np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            current_opponent_progress, _ = project_point_to_centerline(
                np.array([obs["poses_x"][1], obs["poses_y"][1]]), centerline
            )

            if (
                current_ego_progress
                < initial_ego_progress - centerline_total_length / 2
            ):
                current_ego_progress += centerline_total_length
            if (
                current_opponent_progress
                < initial_opponent_progress - centerline_total_length / 2
            ):
                current_opponent_progress += centerline_total_length

            final_state = (
                "overtaking"
                if current_ego_progress > current_opponent_progress
                else "following"
            )

            if np.any(obs["collisions"]):
                done = True
                collision_occurred = True

            elapsed_time = min(
                elapsed_time + timestep, scenario.sim_duration
            )
            while elapsed_time >= next_record_time:
                lidar = downsample_lidar(
                    np.asarray(obs["scans"][0]).ravel(),
                    target_points=End2Race.NUM_LIDAR_FEATURES,
                )
                collected_data.append(
                    [round(next_record_time, 4), ego_steer, ego_speed]
                    + lidar.tolist()
                )
                next_record_time += scenario.sample_interval

            if scenario.render:
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    video_frames.append(frame)

    print("Sim elapsed time:", elapsed_time)

    state_prefix = "o" if final_state == "overtaking" else "f"
    opponent_raceline_number = scenario.opponent_raceline.replace(
        "raceline", ""
    ).replace(".csv", "")
    base_filename = (
        f"{state_prefix}_ol{opponent_raceline_number}_e{scenario.ego_idx}"
        f"_o{opponent_idx}"
        f"_s{scenario.opponent_speed_scale}"
    )

    save_data(
        scenario,
        collected_data,
        video_frames,
        collision_occurred,
        final_state,
        base_filename,
        elapsed_time,
        opponent_idx,
    )

    if scenario.render:
        render_objects = (
            draw_grid_pts
            + draw_traj_pts
            + ego_planner.tracker.drawn_waypoints
        )
        for item in render_objects:
            item.delete()
        draw_grid_pts.clear()
        draw_traj_pts.clear()
        ego_planner.tracker.drawn_waypoints.clear()
        type(env).render_callbacks.clear()
    env.close()


def main():
    require_end2race_runtime()
    if len(sys.argv) != 10:
        raise SystemExit(
            "Usage: python collect.py "
            "<map_name> <dataset_dir> <ego_idx> <opponent_raceline> "
            "<opponent_speed_scale> <sim_duration> <sample_interval> <seed> "
            "<render:true|false>"
        )

    render_value = sys.argv[9]
    if render_value not in {"true", "false"}:
        raise ValueError("render must be true or false")
    scenario = CollectionScenario(
        map_name=sys.argv[1],
        dataset_dir=sys.argv[2],
        ego_idx=int(sys.argv[3]),
        opponent_raceline=sys.argv[4],
        opponent_speed_scale=float(sys.argv[5]),
        sim_duration=float(sys.argv[6]),
        sample_interval=float(sys.argv[7]),
        seed=int(sys.argv[8]),
        render=render_value == "true",
    )
    if (
        not scenario.map_name
        or not scenario.dataset_dir
        or scenario.ego_idx < 0
        or scenario.opponent_speed_scale <= 0
        or scenario.sim_duration <= 0
        or scenario.sample_interval <= 0
    ):
        raise ValueError(
            "map_name and dataset_dir must be nonempty, ego_idx must be "
            "nonnegative, and opponent_speed_scale, sim_duration, and "
            "sample_interval must be positive"
        )

    vehicle = load_project_config().vehicle
    collect_scenario(vehicle, scenario)


if __name__ == "__main__":
    main()
