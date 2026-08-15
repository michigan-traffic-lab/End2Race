import sys
from dataclasses import dataclass
from pathlib import Path

import gym
import imageio
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from config import load_project_config
from expert import create_opponent
from utils import (
    SIMULATION_STEPS_PER_CONTROL,
    EGO_INITIAL_SPEED_FRACTION,
    SIMULATION_TIMESTEP,
    VIDEO_FPS,
    calculate_metrics,
    create_multiagent_render_callback,
    downsample_lidar,
    find_opponent_start_index,
    load_raceline,
    mask_lidar_points,
    project_point_to_centerline,
    require_end2race_runtime,
    unwrap_progress,
)
from model import End2Race

INTERVAL_INDEX = 15
EGO_RACELINE = "raceline1"


@dataclass(frozen=True)
class EvaluationScenario:
    map_name: str
    checkpoint_path: str
    ego_idx: int
    opponent_raceline: str
    opponent_speed_scale: float
    sim_duration: float
    noise: float
    seed: int
    render: bool


def evaluate_segment(model, device, vehicle, scenario):
    rng = np.random.default_rng(scenario.seed)

    ego_waypoints = load_raceline(scenario.map_name, f"{EGO_RACELINE}.csv")
    opp_waypoints = (
        ego_waypoints
        if scenario.opponent_raceline == EGO_RACELINE
        else load_raceline(
            scenario.map_name,
            f"{scenario.opponent_raceline}.csv",
        )
    )
    opp_idx = find_opponent_start_index(
        ego_waypoints,
        opp_waypoints,
        scenario.ego_idx,
        INTERVAL_INDEX,
    )

    normalized_ego_idx = scenario.ego_idx % len(ego_waypoints)
    positions = np.array([
        ego_waypoints[normalized_ego_idx, :3],
        opp_waypoints[opp_idx, :3],
    ])
    initial_speed = EGO_INITIAL_SPEED_FRACTION * vehicle.maximum_speed
    initial_velocities = np.array(
        [
            initial_speed,
            opp_waypoints[opp_idx, 3] * scenario.opponent_speed_scale,
        ]
    )

    env = gym.make(
        "f110-v0",
        map=(
            f"f1tenth_racetracks/{scenario.map_name}/"
            f"{scenario.map_name}_map"
        ),
        map_ext=".png",
        num_agents=2,
        timestep=SIMULATION_TIMESTEP,
        integrator=Integrator.RK4,
    )
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
    if scenario.render:
        render_callback = create_multiagent_render_callback(
            render_info, visited_points, drawn_points, batch_objects
        )
        env.add_render_callback(render_callback)

    video_frames = []
    opponent, _ = create_opponent(
        scenario.map_name, scenario.opponent_raceline
    )
    tracker_steps = opponent.conf.tracker_steps
    hidden_size = model.gru.hidden_size
    hidden_state = torch.zeros((1, 1, hidden_size), device=device)
    previous_speed = initial_speed
    control_step = 0
    ego_steer = 0.0
    ego_speed = initial_speed

    centerline_path = f"f1tenth_racetracks/{scenario.map_name}/raceline1.csv"
    centerline_values = np.loadtxt(centerline_path, delimiter=";", skiprows=1)
    centerline = centerline_values[:, 1:3]
    centerline_total_length = np.linalg.norm(
        np.diff(centerline, axis=0), axis=1
    ).sum()

    obs, _, done, _ = env.reset(
        poses=positions,
        velocities=initial_velocities,
    )

    if scenario.render:
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

    while not done and lap_time < scenario.sim_duration:
        if control_step == 0:
            lidar = mask_lidar_points(
                downsample_lidar(
                    obs["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES
                ),
                scenario.noise,
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
        opponent_speed *= scenario.opponent_speed_scale

        action = np.array(
            [[ego_steer, ego_speed], [opponent_steer, opponent_speed]]
        )
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        control_step = (control_step + 1) % SIMULATION_STEPS_PER_CONTROL

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

        if scenario.render:
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
            frame = env.render(mode="rgb_array")
            if frame is not None:
                video_frames.append(frame)

        if obs["collisions"][0]:
            collision_occurred = True
            done = True

        tracker_count = (tracker_count + 1) % tracker_steps

    if scenario.render and video_frames:
        if collision_occurred:
            state_prefix = "c"
        else:
            state_prefix = "o" if final_state == "overtaking" else "f"
        opponent_raceline_number = scenario.opponent_raceline.replace(
            "raceline", ""
        )
        video_name = (
            f"{state_prefix}_ol{opponent_raceline_number}_e{scenario.ego_idx}"
            f"_o{opp_idx}"
            f"_s{scenario.opponent_speed_scale}.mp4"
        )
        model_name = Path(scenario.checkpoint_path).stem
        noise_suffix = (
            f"_noise{int(scenario.noise * 100)}" if scenario.noise else ""
        )
        video_dir = Path("eval_results") / (
            f"{model_name}_{scenario.map_name}{noise_suffix}"
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / video_name
        imageio.mimwrite(
            video_path, video_frames, fps=VIDEO_FPS, macro_block_size=1
        )
        print(f"Video saved to {video_path}")

    if scenario.render:
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
        "avg_speed": 0 if collision_occurred else avg_speed,
        "speed_variance": 0 if collision_occurred else speed_variance,
        "total_distance": total_distance,
    }


def main():
    require_end2race_runtime()
    if len(sys.argv) != 10:
        raise SystemExit(
            "Usage: python eval_multi.py "
            "<map_name> <checkpoint_path> <ego_idx> <opponent_raceline> "
            "<opponent_speed_scale> <sim_duration> <noise> <seed> "
            "<render:true|false>"
        )

    render_value = sys.argv[9]
    if render_value not in {"true", "false"}:
        raise ValueError("render must be true or false")
    scenario = EvaluationScenario(
        map_name=sys.argv[1],
        checkpoint_path=sys.argv[2],
        ego_idx=int(sys.argv[3]),
        opponent_raceline=sys.argv[4],
        opponent_speed_scale=float(sys.argv[5]),
        sim_duration=float(sys.argv[6]),
        noise=float(sys.argv[7]),
        seed=int(sys.argv[8]),
        render=render_value == "true",
    )
    if (
        not scenario.map_name
        or not scenario.checkpoint_path
        or scenario.ego_idx < 0
        or scenario.opponent_speed_scale <= 0
        or scenario.sim_duration <= 0
        or not 0 <= scenario.noise <= 1
    ):
        raise ValueError(
            "map_name and checkpoint_path must be nonempty, ego_idx must be "
            "nonnegative, opponent_speed_scale and sim_duration must be positive, "
            "and noise must be between 0 and 1"
        )

    vehicle = load_project_config().vehicle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = End2Race().to(device)
    model.load_state_dict(
        torch.load(scenario.checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    result = evaluate_segment(
        model,
        device,
        vehicle,
        scenario,
    )
    print(f"STATE={result['state']}")
    print(f"AVG_SPEED={result['avg_speed']:.3f}")
    print(f"SPEED_VARIANCE={result['speed_variance']:.3f}")
    print(f"TOTAL_DISTANCE={result['total_distance']:.3f}")


if __name__ == "__main__":
    main()
