from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import f110_gym  # Registers the F1TENTH Gym environment.
import gym
import imageio
import numpy as np
import yaml
from f110_gym.envs.base_classes import Integrator

from expert.controllers import RacelineFollower
from expert.lattice_planner import create_expert_planner
from expert.utils import (
    collection_scenarios,
    create_planner_render_callback,
    downsample_lidar,
    find_opponent_start_index,
    project_point_to_centerline,
    raceline_pose,
    unwrap_progress,
    write_collection_summary,
)
from f1tenth_sim.utils import load_simulator_config, simulation_config
from imitation.model import End2Race

VIDEO_OUTPUT_PARAMS = ["-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p"]


def save_data(
    args,
    collected_data,
    video_frames,
    collision_occurred,
    final_state,
    base_filename,
    elapsed_time,
    opponent_idx,
    video_fps,
):
    outcome_dir = "collision" if collision_occurred else "success"
    output_dir = Path(args['dataset_dir']) / outcome_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if collision_occurred:
        collision_metadata = {
            "mode": "multi_agent",
            "ego_raceline": args['ego_raceline'],
            "ego_idx": args['ego_idx'],
            "opp_raceline": args['opponent_raceline'],
            "opp_idx": int(opponent_idx),
            "speed_scale": args['opponent_speed_scale'],
            "interval_idx": args['interval_idx'],
            "simulation_time": float(elapsed_time),
            "final_state": final_state,
        }

        metadata_path = output_dir / f"{base_filename}.json"
        metadata_path.write_text(json.dumps(collision_metadata, indent=2), encoding="utf-8")

        print(f"Collision metadata saved to {metadata_path}")
    else:
        csv_path = output_dir / f"{base_filename}.csv"
        header = ["time", "current_speed", "steer", "desired_speed"] + [
            f"lidar_{index}" for index in range(End2Race.NUM_LIDAR_FEATURES)
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(collected_data)
        print(f"Multi-agent data saved to {csv_path}")
    if video_frames:
        video_path = output_dir / f"{base_filename}.mp4"
        imageio.mimwrite(
            video_path,
            video_frames,
            fps=video_fps,
            codec="libx264",
            macro_block_size=1,
            output_params=VIDEO_OUTPUT_PARAMS,
        )
        print(f"Video saved to {video_path}")


def collect_scenario(args):
    vehicle = load_simulator_config().vehicle
    simulation = simulation_config()
    simulation_steps = round(args['sim_duration'] / simulation.timestep)

    ego_planner = create_expert_planner(args['map_name'], args['ego_raceline'])
    opponent = RacelineFollower(args['map_name'], args['opponent_raceline'])
    expert_tracker_steps = ego_planner.conf.tracker_steps
    env = gym.make(
        "f110-v0",
        map=ego_planner.map_path,
        map_ext=".png",
        timestep=simulation.timestep,
        num_agents=2,
        integrator=Integrator.RK4,
    )

    try:
        if args['render']:
            render_info = {"ego_steer": 0.0, "ego_speed": 0.0, "opp_steer": 0.0, "opp_speed": 0.0}
            draw_traj_pts = []
            env.add_render_callback(
                create_planner_render_callback(render_info, ego_planner, draw_traj_pts)
            )

        ego_waypoints_xytheta = np.column_stack(
            (ego_planner.waypoints[:, :2], ego_planner.waypoints[:, 3])
        )
        ego_position = raceline_pose(ego_waypoints_xytheta, args['ego_idx'])
        opponent_waypoints_xytheta = np.column_stack(
            (opponent.waypoints[:, :2], opponent.waypoints[:, 3])
        )
        opponent_idx = find_opponent_start_index(
            ego_waypoints_xytheta, opponent_waypoints_xytheta, args['ego_idx'], args['interval_idx']
        )
        opponent_position = raceline_pose(opponent_waypoints_xytheta, opponent_idx)
        initial_velocities = np.asarray([
            simulation.ego_initial_speed_fraction * vehicle.maximum_speed,
            opponent.waypoints[opponent_idx, 2] * args['opponent_speed_scale'],
        ])

        # Progress is measured against the ego raceline for both vehicles
        centerline = ego_planner.waypoints[:, :2]
        centerline_total_length = float(
            np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()
        )

        obs, _, done, _ = env.reset(
            poses=np.vstack([ego_position, opponent_position]), velocities=initial_velocities
        )
        if args['render']:
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
            for _ in range(expert_tracker_steps):
                if done or simulation_step >= simulation_steps:
                    break

                if simulation_step % opponent.conf.tracker_steps == 0:
                    opponent_trajectory = opponent.reference_trajectory(
                        obs["poses_x"][1],
                        obs["poses_y"][1],
                    )

                ego_steer, ego_speed = ego_planner.tracker.plan(
                    obs["poses_x"][0],
                    obs["poses_y"][0],
                    obs["poses_theta"][0],
                    obs["linear_vels_x"][0],
                    ego_trajectory,
                )
                ego_steer = np.clip(ego_steer, -vehicle.steering_limit, vehicle.steering_limit)
                opponent_steer, opponent_speed = opponent.tracker.plan(
                    obs["poses_x"][1],
                    obs["poses_y"][1],
                    obs["poses_theta"][1],
                    obs["linear_vels_x"][1],
                    opponent_trajectory,
                )
                opponent_steer = np.clip(
                    opponent_steer, -vehicle.steering_limit, vehicle.steering_limit
                )
                opponent_speed *= args['opponent_speed_scale']
                action = np.asarray([[ego_steer, ego_speed], [opponent_steer, opponent_speed]])

                if args['render']:
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
                final_state = "overtaking" if (
                    current_ego_progress > current_opponent_progress
                ) else "following"

                ego_collision = bool(obs["collisions"][0])
                if ego_collision:
                    done = True
                    collision_occurred = True

                if args['render']:
                    video_frames.append(env.render(mode="rgb_array"))

        elapsed_time = simulation_step * simulation.timestep
        print("Sim elapsed time:", elapsed_time)
        state_prefix = final_state[0]
        opponent_raceline_number = args['opponent_raceline'].removeprefix("raceline")
        base_filename = (
            f"{state_prefix}_ol{opponent_raceline_number}_e{args['ego_idx']}"
            f"_i{args['interval_idx']}_o{opponent_idx}"
            f"_s{args['opponent_speed_scale']}"
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
        return collision_occurred
    finally:
        ego_planner.close()
        env.close()
        if args['render']:
            env_type = type(env.unwrapped)
            env_type.render_callbacks.clear()
            if env_type.renderer is not None:
                env_type.renderer.close()
                env_type.renderer = None


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    settings = config['collection']
    workers = config['runtime']['workers']
    dataset_dir = root / config['paths']['dataset_dir']
    if any(dataset_dir.glob("success/*.csv")) or any(dataset_dir.glob("collision/*.json")):
        raise FileExistsError(f"Collection directory already contains episodes: {dataset_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    scenarios = collection_scenarios(
        settings['map_name'], settings['ego_raceline'], settings['num_startpoints'],
        settings['opponent_racelines'], settings['opponent_speed_scales'],
    )
    print(
        f"Collecting {len(scenarios)} scenarios on {settings['map_name']} "
        f"with {workers} workers"
    )
    completed = collisions = failures = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for start in range(0, len(scenarios), workers):
            pending = {}
            for raceline, scale, ego_idx in scenarios[start:start + workers]:
                job = {**settings, "dataset_dir": dataset_dir,
                       "opponent_raceline": raceline, "opponent_speed_scale": scale,
                       "ego_idx": int(ego_idx)}
                pending[pool.submit(collect_scenario, job)] = (raceline, scale, ego_idx)
            for future in as_completed(pending):
                try:
                    collisions += int(future.result())
                    completed += 1
                except Exception as exc:
                    failures += 1
                    print(f"Scenario {pending[future]} failed: {exc}", flush=True)
            print(
                f"Completed {completed}/{len(scenarios)}; "
                f"collisions={collisions}; errors={failures}",
                flush=True,
            )
    write_collection_summary(dataset_dir, {**settings, "workers": workers}, failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
