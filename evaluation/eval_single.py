import json
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gym
import imageio
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from expert.utils import (
    calculate_metrics,
    create_single_agent_render_callback,
    downsample_lidar,
    project_point_to_centerline,
    unwrap_progress,
)
from expert.lattice_planner import create_expert_planner
from f1tenth_sim.utils import (
    load_simulator_config,
    load_raceline,
    racetrack_path,
    simulation_config,
)
from imitation.model import End2Race

def evaluate_laps(method, model, device, vehicle, args):
    simulation = simulation_config()
    raceline = f"{args['map_name']}_raceline.csv"
    env = gym.make(
        "f110-v0",
        map=str(racetrack_path(args['map_name'], f"{args['map_name']}_map")),
        map_ext=".png",
        num_agents=1,
        timestep=simulation.timestep,
        integrator=Integrator.RK4,
    )

    if args['render']:
        render_info = {"speed": 0.0, "steer": 0.0, "lap_time": 0.0, "laps": 0}
        visited_points = []
        drawn_points = []
        batch_objects = []
        render_callback = create_single_agent_render_callback(
            render_info,
            visited_points,
            drawn_points,
            batch_objects,
            args['lap_num'],
        )
        env.add_render_callback(render_callback)

    waypoints = load_raceline(args['map_name'], raceline)
    start_pose = waypoints[[args['start_idx'] % len(waypoints)], :3]
    initial_speed = simulation.ego_initial_speed_fraction * vehicle.maximum_speed

    centerline = waypoints[:, :2]
    centerline_total_length = np.linalg.norm(
        np.diff(centerline, axis=0), axis=1
    ).sum()

    obs, _, done, _ = env.reset(
        poses=start_pose,
        velocities=np.array([initial_speed]),
    )

    expert_planner = (
        create_expert_planner(args['map_name'], 'raceline1')
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
    expert_tracker_count = 0
    expert_trajectory = None

    initial_progress, _ = project_point_to_centerline(
        np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
    )

    lap_time = 0.0
    collision_occurred = False
    negative_velocity = False
    negative_velocity_value = 0.0
    trajectory = []
    speeds = []
    lap_count = 0
    lap_times = []
    video_frames = []
    lap_start_time = 0.0

    if args['render']:
        env.render("human")

    while not done and lap_count < args['lap_num']:
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

            if ego_speed < 0.0:
                negative_velocity = True
                negative_velocity_value = ego_speed
                done = True
                print(
                    f"Negative desired speed {ego_speed:.6f} m/s at "
                    f"{lap_time:.2f}s"
                )
                break

            ego_steer = np.clip(
                ego_steer, -vehicle.steering_limit, vehicle.steering_limit
            )
            previous_speed = obs["linear_vels_x"][0]

        action = np.array([[ego_steer, ego_speed]])
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        control_step = (control_step + 1) % simulation.steps_per_control
        if method == 'expert':
            expert_tracker_count = (
                expert_tracker_count + 1
            ) % expert_planner.conf.tracker_steps
        current_position = np.array(
            [obs["poses_x"][0], obs["poses_y"][0]]
        )
        trajectory.append(current_position)
        speeds.append(obs["linear_vels_x"][0])

        if args['render']:
            render_info.update(
                {
                    "speed": ego_speed,
                    "steer": ego_steer,
                    "lap_time": lap_time,
                    "laps": lap_count,
                }
            )
            visited_points.append(current_position)

        current_lap_count = int(env.unwrapped.lap_counts[0])
        if current_lap_count > lap_count:
            lap_duration = lap_time - lap_start_time
            lap_start_time = lap_time
            lap_count = current_lap_count
            lap_times.append(lap_duration)
            print(
                f"Lap {lap_count}/{args['lap_num']} completed in "
                f"{lap_duration:.2f}s"
            )
            if lap_count >= args['lap_num']:
                print(f"Successfully completed all {args['lap_num']} laps!")

        if args['render']:
            render_info["laps"] = lap_count

        if obs["collisions"][0]:
            collision_occurred = True
            done = True
            print(f"Wall collision at {lap_time:.2f}s")

        if args['render']:
            video_frames.append(env.render(mode="rgb_array"))

    if trajectory and lap_count < args['lap_num']:
        final_progress, _ = project_point_to_centerline(
            trajectory[-1], centerline
        )
        final_progress = unwrap_progress(
            final_progress,
            initial_progress,
            centerline_total_length,
        )

        lap_fraction = (
            final_progress - initial_progress
        ) / centerline_total_length
        if lap_fraction < 0:
            lap_fraction += 1
        total_lap_progress = lap_count + lap_fraction
    else:
        total_lap_progress = lap_count

    if args['render']:
        for batch_object in batch_objects:
            batch_object.delete()
        type(env.unwrapped).render_callbacks.clear()
        type(env.unwrapped).renderer.close()
        type(env.unwrapped).renderer = None

        if video_frames:
            collision_prefix = "c_" if collision_occurred else ""
            progress_label = f"{total_lap_progress:.1f}".replace(".", "_")
            args['output_dir'].mkdir(parents=True, exist_ok=True)
            video_path = args['output_dir'] / (
                f"{collision_prefix}{args['map_name']}_lap{progress_label}"
                f".mp4"
            )
            imageio.mimwrite(
                video_path, video_frames, fps=simulation.video_fps, macro_block_size=1
            )
            print(f"Video saved to {video_path}")

    env.close()
    if expert_planner is not None:
        expert_planner.close()
    avg_speed, speed_variance, total_distance = calculate_metrics(
        trajectory, speeds
    )
    mean_lap_time = float(np.mean(lap_times)) if lap_times else 0.0

    print("\n" + "=" * 50)
    print("LAP EVALUATION RESULTS")
    print("=" * 50)
    print(f"Map: {args['map_name']}")
    print(f"Target Laps: {args['lap_num']}")
    print(f"Laps Completed: {lap_count}")
    print(f"Lap Progress: {total_lap_progress:.2f} laps")
    print(f"Time Elapsed: {lap_time:.1f}s")
    print(f"Average Speed: {avg_speed:.3f} m/s")
    print(f"Speed Variance: {speed_variance:.3f} m²/s²")
    print(f"Total Distance: {total_distance:.1f} m")

    if lap_times:
        lap_time_variance = np.var(lap_times)
        print(f"\nLap Times: {[f'{t:.2f}s' for t in lap_times]}")
        print(f"Mean Lap Time: {mean_lap_time:.2f}s")
        print(f"Lap Time Variance: {lap_time_variance:.3f}s²")

    passed = (
        not collision_occurred
        and not negative_velocity
        and lap_count >= args['lap_num']
    )
    if collision_occurred:
        print("\nStatus: Collision occurred")
    elif negative_velocity:
        print("\nStatus: Negative desired speed")
    elif passed:
        print("\nStatus: Successfully completed all laps")
    else:
        print("\nStatus: Incomplete - stopped before completing all laps")
    return {
        "passed": passed,
        "collision": collision_occurred,
        "negative_velocity": negative_velocity,
        "negative_velocity_value": float(negative_velocity_value),
        "laps_completed": lap_count,
        "lap_progress": float(total_lap_progress),
        "lap_time": float(lap_time),
        "mean_lap_time": mean_lap_time,
        "avg_speed": float(avg_speed),
        "speed_variance": float(speed_variance),
        "total_distance": float(total_distance),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    args = config['eval_single']
    method = args['method']
    if method not in {'expert', 'bc', 'ppo'}:
        raise ValueError(f"eval_single.method must be expert, bc, or ppo: {method}")
    vehicle = load_simulator_config().vehicle

    model = None
    device = None
    if method != 'expert':
        device = torch.device(config['runtime']['device'])
        model = End2Race().to(device)
        checkpoint_path = root / config['paths']['checkpoint_dir'] / f'{method}.pt'
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model.eval()

    output_dir = root / config['paths']['evaluation_dir'] / method
    results = {}
    for map_name in args['maps']:
        map_args = {**args, "map_name": map_name}
        map_args["output_dir"] = output_dir / map_name
        result = evaluate_laps(method, model, device, vehicle, map_args)
        results[map_name] = result
        print(f"MAP={map_name} PASSED={int(result['passed'])} "
              f"COLLISION={int(result['collision'])} LAPS_COMPLETED={result['laps_completed']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "single.json").write_text(json.dumps(results, indent=2) + "\n")
    raise SystemExit(0 if all(result["passed"] for result in results.values()) else 1)


if __name__ == "__main__":
    main()
