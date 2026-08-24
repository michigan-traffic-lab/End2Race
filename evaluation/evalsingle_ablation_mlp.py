import argparse
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

with redirect_stderr(StringIO()):
    import gym
import imageio
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from expert.utils import (
    calculate_metrics,
    create_single_agent_render_callback,
    downsample_lidar,
    mask_lidar_points,
    project_point_to_centerline,
    require_end2race_runtime,
    unwrap_progress,
)
from f1tenth_sim.utils import (
    load_racetrack_config,
    load_raceline_start,
    racetrack_path,
    simulation_config,
)
from imitation.model_mlp import End2RaceMLP

MINIMUM_LAP_PROGRESS_FRACTION = 0.95


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate End2Race on single-agent laps")
    # Model and artifacts
    parser.add_argument("--map_name", default="Austin")
    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoint/ckp_ablation_mlp/epoch_00500.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results/mlp_ablation"))
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

    centerline = waypoints[:, :2]
    centerline_total_length = np.linalg.norm(
        np.diff(centerline, axis=0), axis=1
    ).sum()

    obs, _, done, _ = env.reset(
        poses=start_pose,
        velocities=np.array([initial_speed]),
    )

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
    simulator_lap_count = 0
    previous_progress = initial_progress
    candidate_lap_progress = 0.0
    lap_times = []
    video_frames = []
    lap_start_time = 0.0

    if args.render:
        env.render("human")

    while not done and lap_count < args.lap_num:
        if control_step == 0:
            lidar = mask_lidar_points(
                downsample_lidar(
                    obs["scans"][0], target_points=End2RaceMLP.NUM_LIDAR_FEATURES
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
                actions = model(lidar_tensor, speed_tensor)
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

        current_progress, _ = project_point_to_centerline(current_position, centerline)
        progress_delta = current_progress - previous_progress
        if progress_delta < -centerline_total_length / 2:
            progress_delta += centerline_total_length
        elif progress_delta > centerline_total_length / 2:
            progress_delta -= centerline_total_length
        candidate_lap_progress += progress_delta
        previous_progress = current_progress

        current_lap_count = int(env.unwrapped.lap_counts[0])
        if current_lap_count > simulator_lap_count:
            simulator_lap_count = current_lap_count
            lap_duration = lap_time - lap_start_time
            valid_lap = (
                progress_delta > 0
                and candidate_lap_progress
                >= MINIMUM_LAP_PROGRESS_FRACTION * centerline_total_length
                and lap_duration > args.minimum_lap_time
            )
            candidate_lap_progress = 0.0
            lap_start_time = lap_time
            if valid_lap:
                lap_count += 1
                lap_times.append(lap_duration)
                print(
                    f"Lap {lap_count}/{args.lap_num} completed in "
                    f"{lap_duration:.2f}s"
                )
                if lap_count >= args.lap_num:
                    print(f"Successfully completed all {args.lap_num} laps!")

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
    model = End2RaceMLP().to(device)
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
