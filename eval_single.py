import argparse
from pathlib import Path

from gym_notices import notices as gym_notices

gym_notices.notices.clear()
import gym
import imageio
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from model import End2Race
from utils import (
    calculate_metrics,
    create_single_agent_render_callback,
    downsample_lidar,
    load_racetrack_config,
    load_raceline_start,
    mask_lidar_points,
    project_point_to_centerline,
    racetrack_path,
    require_end2race_runtime,
    simulation_config,
    unwrap_progress,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate End2Race on single-agent laps")
    # Model and artifacts
    parser.add_argument("--map_name", default="Austin")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--render", action="store_true")

    # Evaluation settings
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lap_num", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--minimum_lap_time", type=float, default=10.0)

    args = parser.parse_args()
    if args.lap_num < 1:
        parser.error("--lap_num must be positive")
    return args


def evaluate_laps(model, device, vehicle, args):
    simulation = simulation_config()
    rng = np.random.default_rng(args.seed)
    raceline = f"{args.map_name}_raceline.csv"
    env = gym.make(
        "f110-v0",
        map=str(racetrack_path(args.map_name, f"{args.map_name}_map")),
        map_ext=".png",
        num_agents=1,
        timestep=simulation.timestep,
        integrator=Integrator.RK4,
    )

    if args.render:
        render_info = {"speed": 0.0, "steer": 0.0, "lap_time": 0.0, "laps": 0}
        visited_points = []
        drawn_points = []
        batch_objects = []
        render_callback = create_single_agent_render_callback(
            render_info,
            visited_points,
            drawn_points,
            batch_objects,
            args.lap_num,
        )
        env.add_render_callback(render_callback)

    start_pose, waypoints = load_raceline_start(
        args.map_name, raceline, args.start_idx
    )
    initial_speed = simulation.ego_initial_speed_fraction * vehicle.maximum_speed
    start_position = start_pose[0, :2]

    centerline = waypoints[:, :2]
    centerline_total_length = np.linalg.norm(
        np.diff(centerline, axis=0), axis=1
    ).sum()

    obs, _, done, _ = env.reset(
        poses=start_pose,
        velocities=np.array([initial_speed]),
    )

    hidden_size = model.gru.hidden_size
    hidden_state = torch.zeros((1, 1, hidden_size), device=device)
    previous_speed = initial_speed
    control_step = 0
    ego_steer = 0.0
    ego_speed = initial_speed

    initial_progress, _ = project_point_to_centerline(
        np.array([obs["poses_x"][0], obs["poses_y"][0]]), centerline
    )

    lap_time = 0.0
    collision_occurred = False
    trajectory = []
    speeds = []
    lap_count = 0
    lap_times = []
    video_frames = []
    near_start_flag = True
    lap_start_time = 0.0

    if args.render:
        env.render("human")

    while not done and lap_count < args.lap_num:
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

        action = np.array([[ego_steer, ego_speed]])
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        control_step = (control_step + 1) % simulation.steps_per_control
        current_position = np.array(
            [obs["poses_x"][0], obs["poses_y"][0]]
        )
        trajectory.append(current_position)
        speeds.append(obs["linear_vels_x"][0])

        if args.render:
            render_info.update(
                {
                    "speed": ego_speed,
                    "steer": ego_steer,
                    "lap_time": lap_time,
                    "laps": lap_count,
                }
            )
            visited_points.append(current_position)

        distance_to_start = np.linalg.norm(current_position - start_position)

        if distance_to_start < 0.5:
            if (
                not near_start_flag
                and lap_time - lap_start_time > args.minimum_lap_time
            ):
                lap_count += 1
                lap_duration = lap_time - lap_start_time
                lap_times.append(lap_duration)
                lap_start_time = lap_time
                print(
                    f"Lap {lap_count}/{args.lap_num} completed in "
                    f"{lap_duration:.2f}s"
                )
                if lap_count >= args.lap_num:
                    print(f"Successfully completed all {args.lap_num} laps!")
            near_start_flag = True
        else:
            near_start_flag = False

        if args.render:
            render_info["laps"] = lap_count

        if obs["collisions"][0]:
            collision_occurred = True
            done = True
            print(f"Wall collision at {lap_time:.2f}s")

        if args.render:
            video_frames.append(env.render(mode="rgb_array"))

    if trajectory and lap_count < args.lap_num:
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

    if args.render:
        for batch_object in batch_objects:
            batch_object.delete()
        type(env.unwrapped).render_callbacks.clear()

        if video_frames:
            collision_prefix = "c_" if collision_occurred else ""
            progress_label = f"{total_lap_progress:.1f}".replace(".", "_")
            noise_suffix = f"_noise{int(args.noise * 100)}" if args.noise else ""
            args.output_dir.mkdir(parents=True, exist_ok=True)
            video_path = args.output_dir / (
                f"{collision_prefix}{args.map_name}_lap{progress_label}"
                f"{noise_suffix}.mp4"
            )
            imageio.mimwrite(
                video_path, video_frames, fps=simulation.video_fps, macro_block_size=1
            )
            print(f"Video saved to {video_path}")

    env.close()
    avg_speed, speed_variance, total_distance = calculate_metrics(
        trajectory, speeds
    )
    mean_lap_time = float(np.mean(lap_times)) if lap_times else 0.0

    print("\n" + "=" * 50)
    print("LAP EVALUATION RESULTS")
    print("=" * 50)
    print(f"Map: {args.map_name}")
    print(f"Target Laps: {args.lap_num}")
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

    passed = not collision_occurred and lap_count >= args.lap_num
    if collision_occurred:
        print("\nStatus: Collision occurred")
    elif passed:
        print("\nStatus: Successfully completed all laps")
    else:
        print("\nStatus: Incomplete - stopped before completing all laps")
    return {
        "passed": passed,
        "collision": collision_occurred,
        "laps_completed": lap_count,
        "lap_progress": float(total_lap_progress),
        "lap_time": float(lap_time),
        "mean_lap_time": mean_lap_time,
        "avg_speed": float(avg_speed),
        "speed_variance": float(speed_variance),
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

    result = evaluate_laps(model, device, vehicle, args)
    print(f"PASSED={int(result['passed'])}")
    print(f"COLLISION={int(result['collision'])}")
    print(f"LAPS_COMPLETED={result['laps_completed']}")
    print(f"LAP_PROGRESS={result['lap_progress']:.3f}")
    print(f"LAP_TIME={result['lap_time']:.3f}")
    print(f"MEAN_LAP_TIME={result['mean_lap_time']:.3f}")
    print(f"AVG_SPEED={result['avg_speed']:.3f}")
    print(f"SPEED_VARIANCE={result['speed_variance']:.3f}")
    print(f"TOTAL_DISTANCE={result['total_distance']:.3f}")

    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
