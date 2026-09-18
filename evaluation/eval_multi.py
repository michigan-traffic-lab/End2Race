from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gym
import imageio
import numpy as np
import torch
import yaml
from f110_gym.envs.base_classes import Integrator

from expert.controllers import RacelineFollower
from expert.lattice_planner import create_expert_planner
from expert.utils import (
    calculate_metrics,
    collection_scenarios,
    create_multiagent_render_callback,
    downsample_lidar,
    find_opponent_start_index,
    project_point_to_centerline,
    unwrap_progress,
)
from f1tenth_sim.utils import (
    load_simulator_config,
    load_raceline,
    racetrack_path,
    simulation_config,
)
from imitation.model import End2Race

def evaluate_segment(method, model, device, vehicle, args):
    simulation = simulation_config()

    ego_waypoints = load_raceline(args['map_name'], f"{args['ego_raceline']}.csv")
    opp_waypoints = (
        ego_waypoints
        if args['opponent_raceline'] == args['ego_raceline']
        else load_raceline(
            args['map_name'],
            f"{args['opponent_raceline']}.csv",
        )
    )
    opp_idx = find_opponent_start_index(
        ego_waypoints,
        opp_waypoints,
        args['ego_idx'],
        args['interval_idx'],
    )

    normalized_ego_idx = args['ego_idx'] % len(ego_waypoints)
    positions = np.array([
        ego_waypoints[normalized_ego_idx, :3],
        opp_waypoints[opp_idx, :3],
    ])
    initial_speed = simulation.ego_initial_speed_fraction * vehicle.maximum_speed
    initial_velocities = np.array(
        [
            initial_speed,
            opp_waypoints[opp_idx, 3] * args['opponent_speed_scale'],
        ]
    )

    env = gym.make(
        "f110-v0",
        map=str(racetrack_path(args['map_name'], f"{args['map_name']}_map")),
        map_ext=".png",
        num_agents=2,
        timestep=simulation.timestep,
        integrator=Integrator.RK4,
    )
    expert_planner = None
    try:
        if args['render']:
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
        opponent = RacelineFollower(args['map_name'], args['opponent_raceline'])
        expert_planner = (
            create_expert_planner(args['map_name'], args['ego_raceline'])
            if method == 'expert'
            else None
        )
        hidden_state = (
            torch.zeros((1, 1, model.gru.hidden_size), device=device)
            if model is not None
            else None
        )
        previous_speed = initial_speed
        control_step = 0
        ego_steer = 0.0
        ego_speed = initial_speed

        centerline_path = racetrack_path(args['map_name'], "raceline1.csv")
        centerline_values = np.loadtxt(centerline_path, delimiter=";", skiprows=1)
        centerline = centerline_values[:, 1:3]
        centerline_total_length = np.linalg.norm(
            np.diff(centerline, axis=0), axis=1
        ).sum()

        obs, _, done, _ = env.reset(
            poses=positions,
            velocities=initial_velocities,
        )

        if args['render']:
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
        opponent_tracker_count = 0
        opponent_trajectory = None
        expert_tracker_count = 0
        expert_trajectory = None

        while not done and lap_time < args['sim_duration']:
            if method == 'expert':
                if expert_tracker_count == 0:
                    expert_trajectory = expert_planner.plan(
                        obs['poses_x'][0],
                        obs['poses_y'][0],
                        obs['poses_theta'][0],
                        obs['scans'][0],
                        obs['linear_vels_x'][0],
                    )
                ego_steer, ego_speed = expert_planner.tracker.plan(
                    obs['poses_x'][0],
                    obs['poses_y'][0],
                    obs['poses_theta'][0],
                    obs['linear_vels_x'][0],
                    expert_trajectory,
                )
                ego_steer = np.clip(
                    ego_steer, -vehicle.steering_limit, vehicle.steering_limit
                )
            elif control_step == 0:
                lidar = downsample_lidar(
                    obs["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES
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

            if opponent_tracker_count == 0:
                opponent_trajectory = opponent.reference_trajectory(
                    obs["poses_x"][1],
                    obs["poses_y"][1],
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
            opponent_speed *= args['opponent_speed_scale']

            action = np.array(
                [[ego_steer, ego_speed], [opponent_steer, opponent_speed]]
            )
            obs, timestep, done, _ = env.step(action)
            lap_time += timestep
            control_step = (control_step + 1) % simulation.steps_per_control
            if method == 'expert':
                expert_tracker_count = (
                    expert_tracker_count + 1
                ) % expert_planner.conf.tracker_steps

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

            if args['render']:
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

            opponent_tracker_count = (
                opponent_tracker_count + 1
            ) % opponent.conf.tracker_steps

        if args['render'] and video_frames:
            state_prefix = "c" if collision_occurred else final_state[0]
            outcome_dir = {
                "c": "collision",
                "f": "follow",
                "o": "overtake",
            }[state_prefix]
            opponent_raceline_number = args['opponent_raceline'].replace("raceline", "")
            video_dir = args['output_dir'] / outcome_dir
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / (
                f"{state_prefix}_ol{opponent_raceline_number}_e{args['ego_idx']}"
                f"_o{opp_idx}_s{args['opponent_speed_scale']}.mp4"
            )
            imageio.mimwrite(
                video_path, video_frames, fps=simulation.video_fps, macro_block_size=1
            )
            print(f"Video saved to {video_path}")

    finally:
        env.close()
        if expert_planner is not None:
            expert_planner.close()
        if args['render']:
            env_type = type(env.unwrapped)
            env_type.render_callbacks.clear()
            if env_type.renderer is not None:
                env_type.renderer.close()
                env_type.renderer = None
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


def run_evaluation(args, device):
    vehicle = load_simulator_config().vehicle
    method = args['method']

    if method == 'expert':
        return evaluate_segment(method, None, None, vehicle, args)

    device = torch.device(device)
    model = End2Race().to(device)
    model.load_state_dict(
        torch.load(args['checkpoint_path'], map_location=device, weights_only=True)
    )
    model.eval()

    return evaluate_segment(method, model, device, vehicle, args)


def write_summary(settings, map_name, records, planned, output_dir, stop_reason):
    counts = {
        "following": sum(record['state'] == 1 for record in records),
        "overtaking": sum(record['state'] == 2 for record in records),
        "collision": sum(record['state'] == 3 for record in records),
        "errors": sum(record['state'] == 0 for record in records),
    }
    counts['success'] = counts['following'] + counts['overtaking']
    completed = len(records)
    summary = {
        **{key: value for key, value in settings.items() if key != 'maps'},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map_name": map_name,
        "planned_scenarios": planned,
        "completed_scenarios": completed,
        "complete": completed == planned and stop_reason == "completed",
        "stop_reason": stop_reason,
        **counts,
    }
    for name in ('following', 'overtaking', 'success', 'collision'):
        summary[f'{name}_percent'] = round(100 * counts[name] / completed, 4) if completed else 0.0
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{map_name}: {completed}/{planned} completed; "
          f"following={counts['following']}, overtaking={counts['overtaking']}, "
          f"collisions={counts['collision']}, errors={counts['errors']}", flush=True)
    return summary['complete'] and counts['errors'] == 0


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    settings = config['eval_multi']
    method = settings['method']
    if method not in {'expert', 'bc', 'ppo'}:
        raise ValueError(f"eval_multi.method must be expert, bc, or ppo: {method}")
    workers = config['runtime']['workers']
    checkpoint_path = root / config['paths']['checkpoint_dir'] / f'{method}.pt'
    output_root = root / config['paths']['evaluation_dir'] / method
    complete = True
    for map_name in settings['maps']:
        output_dir = output_root / map_name
        output_dir.mkdir(parents=True, exist_ok=True)
        scenarios = collection_scenarios(
            map_name, settings['ego_raceline'], settings['num_startpoints'],
            settings['opponent_racelines'], settings['opponent_speed_scales'],
        )
        print(
            f"Evaluating {len(scenarios)} scenarios on {map_name} "
            f"with {workers} workers",
            flush=True,
        )
        records = []
        stop_reason = "completed"
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                for start in range(0, len(scenarios), workers):
                    pending = {}
                    for raceline, scale, ego_idx in scenarios[start:start + workers]:
                        job = {**settings, "map_name": map_name, "output_dir": output_dir,
                               "checkpoint_path": checkpoint_path,
                               "opponent_raceline": raceline, "opponent_speed_scale": scale,
                               "ego_idx": int(ego_idx)}
                        future = pool.submit(run_evaluation, job, config['runtime']['device'])
                        pending[future] = (raceline, scale, ego_idx)
                    for future in as_completed(pending):
                        try:
                            records.append(future.result())
                        except Exception as exc:
                            records.append({"state": 0})
                            print(
                                f"{map_name} scenario {pending[future]} failed: {exc}",
                                flush=True,
                            )
                    print(f"{map_name}: {len(records)}/{len(scenarios)} finished", flush=True)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        finally:
            map_complete = write_summary(
                settings,
                map_name,
                records,
                len(scenarios),
                output_dir,
                stop_reason,
            )
        complete = complete and map_complete
        if stop_reason == "interrupted":
            raise SystemExit(130)
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
