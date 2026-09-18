import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numba import njit

import yaml
from f1tenth_sim.utils import (
    load_simulator_config,
    racetrack_path,
    simulation_config,
)


def expert_configuration():
    with (Path(__file__).resolve().parents[1] / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    return SimpleNamespace(
        **config['expert'],
        parallel_workers=config['runtime']['workers'],
        **vars(load_simulator_config().vehicle),
    )


def opponent_configuration():
    with (Path(__file__).resolve().parents[1] / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    return SimpleNamespace(
        **config['opponent'],
        **vars(load_simulator_config().vehicle),
    )


@njit(cache=True)
def nearest_point(point, trajectory):
    """Project a point onto an open piecewise-linear trajectory."""
    differences = trajectory[1:] - trajectory[:-1]
    squared_lengths = differences[:, 0] ** 2 + differences[:, 1] ** 2
    projections = np.empty_like(differences)
    fractions = np.empty(len(differences))
    distances = np.empty(len(differences))
    for index in range(len(differences)):
        fraction = np.dot(point - trajectory[index], differences[index])
        fraction /= squared_lengths[index]
        fraction = min(max(fraction, 0.0), 1.0)
        fractions[index] = fraction
        projections[index] = trajectory[index] + fraction * differences[index]
        offset = point - projections[index]
        distances[index] = np.sqrt(np.dot(offset, offset))
    segment_index = np.argmin(distances)
    return (
        projections[segment_index],
        distances[segment_index],
        fractions[segment_index],
        segment_index,
    )


@njit(cache=True)
def project_point_to_centerline(point, centerline):
    """Return distance travelled along a centerline and its nearest segment."""
    _, _, fraction, segment_index = nearest_point(point, centerline)
    progress = 0.0
    for index in range(segment_index):
        progress += np.linalg.norm(centerline[index + 1] - centerline[index])
    progress += fraction * np.linalg.norm(
        centerline[segment_index + 1] - centerline[segment_index]
    )
    return progress, segment_index


def downsample_lidar(lidar_data, target_points):
    """Uniformly downsample a flat LiDAR scan."""
    scan = np.asarray(lidar_data).reshape(-1)
    if scan.size % target_points == 0:
        step = scan.size // target_points
        return scan[::step][:target_points]
    indices = np.linspace(0, scan.size - 1, target_points, dtype=np.int64)
    return scan[indices]


def find_opponent_start_index(ego_waypoints, opponent_waypoints, ego_idx, interval_idx):
    """Map an ego start onto another raceline and apply a waypoint gap."""
    ego_waypoints = ego_waypoints[:-1]
    opponent_waypoints = opponent_waypoints[:-1]
    ego_waypoint = ego_waypoints[ego_idx % len(ego_waypoints)]
    distances = np.linalg.norm(opponent_waypoints[:, :2] - ego_waypoint[:2], axis=1)
    return (int(np.argmin(distances)) + interval_idx) % len(opponent_waypoints)


def unwrap_progress(progress, initial_progress, track_length):
    """Keep progress continuous when a vehicle crosses the lap boundary."""
    if progress < initial_progress - track_length / 2:
        return progress + track_length
    return progress


def raceline_pose(waypoints_xytheta, index):
    """Return one pose from a periodic raceline."""
    pose = np.array(
        waypoints_xytheta[index % len(waypoints_xytheta), :3],
        copy=True,
    )
    pose[2] %= 2.0 * np.pi
    return pose


def calculate_metrics(trajectory, speeds):
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    total_distance = sum(
        np.linalg.norm(np.asarray(trajectory[index + 1]) - trajectory[index])
        for index in range(len(trajectory) - 1)
    )
    return avg_speed, speed_variance, total_distance


def follow_vehicle_camera(event, horizontal_margin=800.0):
    """Center the camera on the ego vehicle at the normal render scale."""
    x_vertices = event.cars[0].vertices[::2]
    y_vertices = event.cars[0].vertices[1::2]
    center_x = float(np.mean(x_vertices))
    center_y = float(np.mean(y_vertices))
    width, height = event.get_size()
    vertical_margin = horizontal_margin * height / width
    event.left, event.right = center_x - horizontal_margin, center_x + horizontal_margin
    event.top, event.bottom = center_y + vertical_margin, center_y - vertical_margin


def position_score_label(event):
    event.score_label.x = event.left + 8.0
    event.score_label.y = event.top - 8.0


def update_point_batches(
    event, batches, points, color, batch_objects=None, scale=10.0
):
    """Populate or update pyglet point batches with the provided 2D points."""
    from pyglet.gl import GL_POINTS

    color_stream = list(color)
    for idx, point in enumerate(points):
        x_coord, y_coord = float(point[0]) * scale, float(point[1]) * scale
        if idx < len(batches):
            batches[idx].vertices = [x_coord, y_coord, 0.0]
        else:
            batch_item = event.batch.add(
                1,
                GL_POINTS,
                None,
                ("v3f/stream", [x_coord, y_coord, 0.0]),
                ("c3B/stream", color_stream),
            )
            batches.append(batch_item)
            if batch_objects is not None:
                batch_objects.append(batch_item)


def create_multiagent_render_callback(
    render_info, visited_points, drawn_points, batch_objects
):
    """Create a render callback that visualizes two vehicles and their trajectories."""
    colors = [(48, 112, 162), (193, 82, 75)]

    def render_callback(event):
        follow_vehicle_camera(event)
        position_score_label(event)

        event.score_label.text = (
            f"State: {render_info['state']} | "
            f"Ego: {render_info['ego_speed']:.1f}m/s, "
            f"{render_info['ego_steer']:+.2f}rad | "
            f"Opp: {render_info['opp_speed']:.1f}m/s, "
            f"{render_info['opp_steer']:+.2f}rad"
        )

        for vehicle_idx, color in enumerate(colors):
            update_point_batches(
                event,
                drawn_points[vehicle_idx],
                visited_points[vehicle_idx],
                color,
                batch_objects=batch_objects,
                scale=50.0,
            )

    return render_callback


def create_planner_render_callback(render_info, planner, draw_traj_pts):
    def render_callback(event):
        follow_vehicle_camera(event)
        position_score_label(event)

        event.score_label.text = (
            f"Ego: {render_info['ego_speed']:.1f}m/s, "
            f"{render_info['ego_steer']:+.2f}rad | "
            f"Opp: {render_info['opp_speed']:.1f}m/s, "
            f"{render_info['opp_steer']:+.2f}rad"
        )

        if planner.best_trajectory is not None:
            update_point_batches(
                event,
                draw_traj_pts,
                planner.best_trajectory[:, :2],
                color=(183, 193, 222),
                scale=50.0,
            )

    return render_callback


def create_single_agent_render_callback(
    render_info, visited_points, drawn_points, batch_objects, lap_num
):
    def render_callback(event):
        follow_vehicle_camera(event)
        position_score_label(event)

        event.score_label.text = (
            f"Laps: {render_info['laps']}/{lap_num} | "
            f"Time: {render_info['lap_time']:.1f}s | "
            f"Speed: {render_info['speed']:.1f}m/s | "
            f"Steer: {render_info['steer']:+.2f}rad"
        )

        update_point_batches(
            event,
            drawn_points,
            visited_points,
            color=(48, 112, 162),
            batch_objects=batch_objects,
            scale=50.0,
        )

    return render_callback


def get_ego_idx_range(map_name, ego_raceline, num_startpoints):
    raceline_path = racetrack_path(map_name, f"{ego_raceline}.csv")
    waypoints = np.loadtxt(raceline_path, delimiter=";", skiprows=1, ndmin=2)
    unique_waypoints = waypoints[:-1]
    track_length = waypoints[-1, 0]
    targets = np.arange(num_startpoints) * track_length / num_startpoints
    progress_delta = np.abs(unique_waypoints[:, None, 0] - targets[None, :])
    progress_delta = np.minimum(progress_delta, track_length - progress_delta)
    return np.argmin(progress_delta, axis=0).astype(int).tolist()


def collection_scenarios(map_name, ego_raceline, num_startpoints, opponent_racelines, opponent_speed_scales):
    """Every (opponent raceline, speed scale, ego index) triple in one collection."""
    ego_indices = get_ego_idx_range(map_name, ego_raceline, num_startpoints)
    return [
        (raceline, speed_scale, ego_idx)
        for raceline in opponent_racelines
        for speed_scale in opponent_speed_scales
        for ego_idx in ego_indices
    ]


def _episode_record(path, outcome):
    """Recover one scenario's parameters from its artifact filename."""
    state, raceline, ego_idx, _, _, speed_scale = path.stem.split("_")
    return {
        "outcome": outcome,
        "final_state": "overtaking" if state == "o" else "following",
        "opponent_raceline": f"raceline{raceline[2:]}",
        "ego_idx": int(ego_idx[1:]),
        "speed_scale": float(speed_scale[1:]),
    }


def _percentage(numerator, denominator):
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _tally(episodes):
    collision_free = [item for item in episodes if item["outcome"] == "collision_free"]
    overtakes = sum(item["final_state"] == "overtaking" for item in collision_free)
    return {
        "recorded_scenarios": len(episodes),
        "collision_free_scenarios": len(collision_free),
        "collision_scenarios": len(episodes) - len(collision_free),
        "successful_overtakes": overtakes,
        "collision_free_following": len(collision_free) - overtakes,
        "collision_free_rate_percent": _percentage(len(collision_free), len(episodes)),
        "successful_overtake_rate_percent": _percentage(overtakes, len(episodes)),
    }


def write_collection_summary(dataset_dir, collection_config, failures):
    """Summarize a finished collection from the artifacts left in its dataset directory."""
    with (Path(__file__).resolve().parents[1] / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    # Imported here so that every utils consumer does not pay for torch
    from imitation.model import End2Race

    simulation = simulation_config()
    dataset_dir = Path(dataset_dir)
    success_dir = dataset_dir / "success"
    collision_dir = dataset_dir / "collision"
    success_paths = list(success_dir.glob("*.csv"))
    collision_paths = list(collision_dir.glob("*.json"))

    episodes = [_episode_record(path, "collision_free") for path in success_paths] + [
        _episode_record(path, "collision") for path in collision_paths
    ]
    results = _tally(episodes)
    opponent_racelines = collection_config["opponent_racelines"]
    opponent_speed_scales = collection_config["opponent_speed_scales"]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "collection_config": {
            "mode": "multi_agent",
            **collection_config,
            "ego_indices": sorted({item["ego_idx"] for item in episodes}),
            "simulation_frequency_hz": simulation.frequency_hz,
            "control_frequency_hz": simulation.control_frequency_hz,
            "expert_planner_frequency_hz": simulation.expert_planner_frequency_hz,
        },
        "data_config": {
            "lidar_features": End2Race.NUM_LIDAR_FEATURES,
            "csv_columns": 4 + End2Race.NUM_LIDAR_FEATURES,
            "vehicle": vars(load_simulator_config().vehicle),
            "expert": config['expert'],
        },
        "results": {
            "expected_scenarios": collection_config["num_startpoints"]
            * len(opponent_racelines)
            * len(opponent_speed_scales),
            "collection_process_failures": failures,
            **results,
            "collisions_while_overtaking": sum(
                item["final_state"] == "overtaking"
                for item in episodes
                if item["outcome"] == "collision"
            ),
            "training_rows": sum(
                sum(1 for _ in path.open(encoding="utf-8")) - 1 for path in success_paths
            ),
            "success_videos": len(list(success_dir.glob("*.mp4"))),
            "collision_videos": len(list(collision_dir.glob("*.mp4"))),
            "breakdown": [
                {
                    "opponent_raceline": raceline,
                    "opponent_speed_scale": speed_scale,
                    **_tally([
                        item
                        for item in episodes
                        if item["opponent_raceline"] == raceline and item["speed_scale"] == speed_scale
                    ]),
                }
                for raceline in opponent_racelines
                for speed_scale in opponent_speed_scales
            ],
        },
    }

    summary_path = dataset_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Collection finished")
    print(f"following: {results['collision_free_following']}")
    print(f"overtaking: {results['successful_overtakes']}")
    print(f"collisions: {results['collision_scenarios']}")
    print(f"failures: {failures}")
    print(f"Dataset summary saved to {summary_path}")
    return summary
