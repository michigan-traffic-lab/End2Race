import os
import sys

import numpy as np
from numba import njit

SIMULATION_TIMESTEP = 0.01
VIDEO_FPS = 100
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


def random_position(
    waypoints_xytheta,
    sampled_number=1,
    rng=None,
    xy_noise=0.0,
    theta_noise=0.0,
    ego_idx=100,
    interval_idx=20,
):
    """Return deterministic or noise-perturbed starting poses on a raceline."""
    if rng is None:
        rng = np.random.default_rng()
    poses = []
    for sample_index in range(sampled_number):
        waypoint_index = (
            ego_idx + sample_index * interval_idx
        ) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[waypoint_index, :3]
        x += rng.random() * xy_noise
        y += rng.random() * xy_noise
        theta = (theta % (2.0 * np.pi)) + rng.random() * theta_noise
        poses.append((x, y, theta))
    return np.asarray(poses), ego_idx


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
    waypoint_count = len(np.loadtxt(raceline_path, delimiter=";", skiprows=1))
    return np.linspace(
        0, waypoint_count - 1, num_startpoints, dtype=int
    ).tolist()
