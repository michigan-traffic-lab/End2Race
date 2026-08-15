import argparse
from dataclasses import dataclass
from pathlib import Path

import gym
import imageio
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from config import load_project_config
from model import End2Race
from utils import (
    SIMULATION_STEPS_PER_CONTROL,
    EGO_INITIAL_SPEED_FRACTION,
    SIMULATION_TIMESTEP,
    VIDEO_FPS,
    calculate_metrics,
    create_single_agent_render_callback,
    downsample_lidar,
    load_raceline_start,
    mask_lidar_points,
    project_point_to_centerline,
    require_end2race_runtime,
    unwrap_progress,
)

EVALUATION_NOISE = 0.0
EVALUATION_SEED = 42
LAP_COUNT = 1
START_INDEX = 0
MINIMUM_LAP_TIME = 10.0


@dataclass(frozen=True)
class EvaluationSettings:
    map_name: str
    checkpoint_path: Path
    noise: float
    seed: int
    render: bool
    lap_num: int
    start_idx: int
    minimum_lap_time: float


def evaluate_laps(model, device, vehicle, settings):
    rng = np.random.default_rng(settings.seed)
    raceline = f"{settings.map_name}_raceline.csv"
    env = gym.make(
        "f110-v0",
        map=(
            f"f1tenth_racetracks/{settings.map_name}/"
            f"{settings.map_name}_map"
        ),
        map_ext=".png",
        num_agents=1,
        timestep=SIMULATION_TIMESTEP,
        integrator=Integrator.RK4,
    )

    video_path = None
    if settings.render:
        model_name = Path(settings.checkpoint_path).stem
        noise_str = (
            f"_noise{int(settings.noise * 100)}" if settings.noise > 0 else ""
        )
        video_dir = Path("eval_results") / f"{model_name}{noise_str}"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / (
            f"{model_name}_{settings.map_name}{noise_str}.mp4"
        )

    render_info = {"speed": 0.0, "steer": 0.0, "lap_time": 0.0, "laps": 0}
    visited_points = []
    drawn_points = []
    batch_objects = []
    if settings.render:
        render_callback = create_single_agent_render_callback(
            render_info,
            visited_points,
            drawn_points,
            batch_objects,
            settings.lap_num,
        )
        env.add_render_callback(render_callback)

    start_pose, waypoints = load_raceline_start(
        settings.map_name, raceline, settings.start_idx
    )
    initial_speed = EGO_INITIAL_SPEED_FRACTION * vehicle.maximum_speed
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

    if settings.render:
        env.render("human")

    while not done and lap_count < settings.lap_num:
        if control_step == 0:
            lidar = mask_lidar_points(
                downsample_lidar(
                    obs["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES
                ),
                settings.noise,
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
        control_step = (control_step + 1) % SIMULATION_STEPS_PER_CONTROL
        current_position = np.array(
            [obs["poses_x"][0], obs["poses_y"][0]]
        )
        trajectory.append(current_position)
        speeds.append(obs["linear_vels_x"][0])

        if settings.render:
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
                and lap_time - lap_start_time > settings.minimum_lap_time
            ):
                lap_count += 1
                lap_duration = lap_time - lap_start_time
                lap_times.append(lap_duration)
                lap_start_time = lap_time
                print(
                    f"Lap {lap_count}/{settings.lap_num} completed in "
                    f"{lap_duration:.2f}s"
                )
                if lap_count >= settings.lap_num:
                    print(f"Successfully completed all {settings.lap_num} laps!")
            near_start_flag = True
        else:
            near_start_flag = False

        if settings.render:
            render_info["laps"] = lap_count

        if obs["collisions"][0]:
            collision_occurred = True
            done = True
            print(f"Wall collision at {lap_time:.2f}s")

        if settings.render:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                video_frames.append(frame)

    if trajectory and lap_count < settings.lap_num:
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

    if settings.render:
        for batch_object in batch_objects:
            batch_object.delete()
        type(env.unwrapped).render_callbacks.clear()

        if video_frames:
            imageio.mimwrite(
                video_path, video_frames, fps=VIDEO_FPS, macro_block_size=1
            )
            print(f"Video saved to {video_path}")

    env.close()
    avg_speed, speed_variance, total_distance = calculate_metrics(
        trajectory, speeds
    )

    print("\n" + "=" * 50)
    print("LAP EVALUATION RESULTS")
    print("=" * 50)
    print(f"Map: {settings.map_name}")
    print(f"Target Laps: {settings.lap_num}")
    print(f"Laps Completed: {lap_count}")
    print(f"Lap Progress: {total_lap_progress:.2f} laps")
    print(f"Time Elapsed: {lap_time:.1f}s")
    print(f"Average Speed: {avg_speed:.3f} m/s")
    print(f"Speed Variance: {speed_variance:.3f} m²/s²")
    print(f"Total Distance: {total_distance:.1f} m")

    if lap_times:
        mean_lap_time = np.mean(lap_times)
        lap_time_variance = np.var(lap_times) if len(lap_times) > 1 else 0
        print(f"\nLap Times: {[f'{t:.2f}s' for t in lap_times]}")
        print(f"Mean Lap Time: {mean_lap_time:.2f}s")
        print(f"Lap Time Variance: {lap_time_variance:.3f}s²")

    passed = not collision_occurred and lap_count >= settings.lap_num
    if collision_occurred:
        print("\nStatus: Collision occurred")
    elif passed:
        print("\nStatus: Successfully completed all laps")
    else:
        print("\nStatus: Incomplete - stopped before completing all laps")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_name")
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--render", action="store_true")
    arguments = parser.parse_args()

    require_end2race_runtime()
    project = load_project_config()
    checkpoint_path = arguments.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    settings = EvaluationSettings(
        map_name=arguments.map_name,
        checkpoint_path=checkpoint_path,
        noise=EVALUATION_NOISE,
        seed=EVALUATION_SEED,
        render=arguments.render,
        lap_num=LAP_COUNT,
        start_idx=START_INDEX,
        minimum_lap_time=MINIMUM_LAP_TIME,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = End2Race().to(device)
    model.load_state_dict(
        torch.load(settings.checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    raise SystemExit(
        0
        if evaluate_laps(model, device, project.vehicle, settings)
        else 1
    )


if __name__ == "__main__":
    main()
