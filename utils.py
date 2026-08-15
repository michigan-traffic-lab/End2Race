import os
import sys

import numpy as np
from numba import njit

SIMULATION_FREQUENCY_HZ = 120
CONTROL_FREQUENCY_HZ = 40
EXPERT_PLANNER_FREQUENCY_HZ = 10
if (
    SIMULATION_FREQUENCY_HZ % CONTROL_FREQUENCY_HZ
    or SIMULATION_FREQUENCY_HZ % EXPERT_PLANNER_FREQUENCY_HZ
):
    raise ValueError(
        "Simulation frequency must divide both control and expert planner "
        "frequencies"
    )
SIMULATION_STEPS_PER_CONTROL = (
    SIMULATION_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ
)
SIMULATION_STEPS_PER_EXPERT_PLAN = (
    SIMULATION_FREQUENCY_HZ // EXPERT_PLANNER_FREQUENCY_HZ
)
SIMULATION_TIMESTEP = 1.0 / SIMULATION_FREQUENCY_HZ
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY_HZ
VIDEO_FPS = SIMULATION_FREQUENCY_HZ
EGO_INITIAL_SPEED_FRACTION = 0.5


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
    if target_points <= 0:
        raise ValueError("target_points must be positive")
    if scan.size < target_points:
        raise ValueError(
            f"Cannot downsample {scan.size} LiDAR points to {target_points}"
        )
    if scan.size == target_points:
        return scan.copy()
    if scan.size % target_points == 0:
        step = scan.size // target_points
        return scan[::step][:target_points]
    indices = np.linspace(0, scan.size - 1, target_points, dtype=np.int64)
    return scan[indices]


def find_corresponding_waypoint(ego_waypoint, opponent_waypoints):
    """Find the opponent-raceline waypoint nearest to an ego waypoint."""
    distances = np.linalg.norm(
        opponent_waypoints[:, :2] - ego_waypoint[:2], axis=1
    )
    return int(np.argmin(distances))


def find_opponent_start_index(
    ego_waypoints,
    opponent_waypoints,
    ego_idx,
    interval_idx,
):
    """Map an ego start onto another raceline and apply a waypoint gap."""
    ego_waypoints = ego_waypoints[:-1]
    opponent_waypoints = opponent_waypoints[:-1]
    normalized_ego_idx = ego_idx % len(ego_waypoints)
    mapped_idx = find_corresponding_waypoint(
        ego_waypoints[normalized_ego_idx],
        opponent_waypoints,
    )
    return (mapped_idx + interval_idx) % len(opponent_waypoints)


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


def require_end2race_runtime():
    """Require the supported Python 3.11 end2race Conda environment."""
    environment_name = os.environ.get("CONDA_DEFAULT_ENV") or os.path.basename(
        sys.prefix
    )
    if sys.version_info[:2] != (3, 11) or environment_name != "end2race":
        raise RuntimeError(
            "Activate the Python 3.11 end2race environment before running this "
            "workflow: conda activate end2race"
        )


def load_raceline(map_name, raceline_file):
    """Load x, y, heading, and speed columns from a raceline."""
    raceline_path = os.path.join("f1tenth_racetracks", map_name, raceline_file)
    values = np.loadtxt(raceline_path, delimiter=";", skiprows=1, ndmin=2)
    if values.shape[1] < 6:
        raise ValueError(f"{raceline_path} must contain at least six columns")
    return values[:, [1, 2, 3, 5]]


def load_raceline_start(map_name, raceline_file, start_idx):
    waypoints = load_raceline(map_name, raceline_file)
    idx = start_idx % len(waypoints)
    start_pose = np.array(
        [[waypoints[idx, 0], waypoints[idx, 1], waypoints[idx, 2]]]
    )
    return start_pose, waypoints


def calculate_metrics(trajectory, speeds):
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    total_distance = (
        sum(
            np.linalg.norm(np.asarray(trajectory[index + 1]) - trajectory[index])
            for index in range(len(trajectory) - 1)
        )
        if len(trajectory) > 1
        else 0
    )
    return avg_speed, speed_variance, total_distance


def mask_lidar_points(lidar, ratio, rng):
    lidar = np.array(lidar, copy=True)
    masked_count = int(len(lidar) * ratio)
    if masked_count:
        indices = rng.choice(len(lidar), masked_count, replace=False)
        lidar[indices] = 0.0
    return lidar


def follow_vehicle_camera(event, margin=800.0):
    """Center the camera on the specified vehicle and apply symmetric margins."""
    x_vertices = event.cars[0].vertices[::2]
    y_vertices = event.cars[0].vertices[1::2]
    center_x = float(np.mean(x_vertices))
    center_y = float(np.mean(y_vertices))
    event.left, event.right = center_x - margin, center_x + margin
    event.top, event.bottom = center_y + margin, center_y - margin


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
    colors = [(255, 255, 0), (255, 0, 0)]

    def render_callback(event):
        follow_vehicle_camera(event)
        event.score_label.x = event.left + 800
        event.score_label.y = event.bottom + 100

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
        event.score_label.x = event.left + 800
        event.score_label.y = event.bottom + 100

        event.score_label.text = (
            f"Ego: {render_info['ego_speed']:.1f}m/s, "
            f"{render_info['ego_steer']:+.2f}rad | "
            f"Opp: {render_info['opp_speed']:.1f}m/s, "
            f"{render_info['opp_steer']:+.2f}rad"
        )

        if planner.best_trajectory is not None:
            trajectory_points = planner.best_trajectory[:, :2]
            update_point_batches(
                event,
                draw_traj_pts,
                trajectory_points,
                color=(183, 193, 222),
                scale=50.0,
            )

    return render_callback


def create_single_agent_render_callback(
    render_info, visited_points, drawn_points, batch_objects, lap_num
):
    def render_callback(event):
        follow_vehicle_camera(event)
        event.score_label.x = event.left + 800
        event.score_label.y = event.top - 1500

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
            color=(255, 255, 0),
            batch_objects=batch_objects,
            scale=50.0,
        )

    return render_callback


def get_ego_idx_range(map_name, ego_raceline, num_startpoints):
    raceline_path = os.path.join(
        "f1tenth_racetracks", map_name, f"{ego_raceline}.csv"
    )
    waypoints = np.loadtxt(raceline_path, delimiter=";", skiprows=1, ndmin=2)
    if np.linalg.norm(waypoints[-1, 1:3] - waypoints[0, 1:3]) > 1e-9:
        raise ValueError(f"{raceline_path} must repeat its first waypoint at the end")
    unique_waypoints = waypoints[:-1]
    track_length = waypoints[-1, 0]
    targets = np.arange(num_startpoints) * track_length / num_startpoints
    progress_delta = np.abs(unique_waypoints[:, None, 0] - targets[None, :])
    progress_delta = np.minimum(progress_delta, track_length - progress_delta)
    return np.argmin(progress_delta, axis=0).astype(int).tolist()
