import os
import sys

import numpy as np

SIMULATION_TIMESTEP = 0.01
VIDEO_FPS = 100


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


def load_raceline_with_speed(map_name, raceline_file, start_idx):
    waypoints = load_raceline(map_name, raceline_file)
    idx = start_idx % len(waypoints)
    start_pose = np.array(
        [[waypoints[idx, 0], waypoints[idx, 1], waypoints[idx, 2]]]
    )
    initial_speed = waypoints[idx, 3]
    return start_pose, initial_speed, waypoints


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


def create_planner_render_callback(
    render_info, planner, draw_grid_pts, draw_traj_pts
):

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

        if planner.goal_grid is not None:
            goal_grid_pts = np.column_stack(
                (planner.goal_grid[:, 0], planner.goal_grid[:, 1])
            )
            update_point_batches(
                event,
                draw_grid_pts,
                goal_grid_pts,
                color=(183, 193, 222),
                scale=50.0,
            )

            if planner.best_traj is not None:
                best_traj_pts = np.column_stack(
                    (planner.best_traj[:, 0], planner.best_traj[:, 1])
                )
                update_point_batches(
                    event,
                    draw_traj_pts,
                    best_traj_pts,
                    color=(183, 193, 222),
                    scale=50.0,
                )

        planner.tracker.render_waypoints(event)

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
