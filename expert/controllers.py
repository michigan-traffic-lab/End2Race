import math

import numpy as np

from expert.utils import expert_configuration, nearest_point
from f1tenth_sim.utils import racetrack_path


class PurePursuitController:
    """Track planned trajectories with pure pursuit."""

    def __init__(self, configuration):
        self.min_lookahead = configuration.min_lookahead
        self.max_lookahead = configuration.max_lookahead
        self.lookahead_speed_scale = configuration.lookahead_speed_scale
        self.steering_gain = configuration.steering_gain
        self.interpolation_points = configuration.interpolation_points

    def plan(self, pose_x, pose_y, pose_theta, current_speed, trajectory):
        lookahead = (
            current_speed
            * (self.max_lookahead - self.min_lookahead)
            / self.lookahead_speed_scale
            + self.min_lookahead
        )
        position = np.array([pose_x, pose_y])
        distances = np.linalg.norm(trajectory[:, :2] - position, axis=1)
        segment_end = int(np.argmin(distances))
        if distances[-1] < lookahead:
            segment_end = len(trajectory) - 1
        else:
            while segment_end + 1 < len(trajectory) and distances[segment_end] < lookahead:
                segment_end += 1
        segment_start = max(segment_end - 1, 0)
        x_values = np.linspace(trajectory[segment_start, 0], trajectory[segment_end, 0], self.interpolation_points)
        y_values = np.linspace(trajectory[segment_start, 1], trajectory[segment_end, 1], self.interpolation_points)
        speed_values = np.linspace(trajectory[segment_start, 2], trajectory[segment_end, 2], self.interpolation_points)
        interpolated = np.column_stack((x_values, y_values))
        index = int(np.argmin(np.abs(np.linalg.norm(interpolated - position, axis=1) - lookahead)))
        target = interpolated[index]
        actual_lookahead = max(np.linalg.norm(position - target), 1e-6)
        lateral_error = np.dot(
            np.array([math.sin(-pose_theta), math.cos(-pose_theta)]), target - position
        )
        error = 2.0 * lateral_error / actual_lookahead**2
        steering = self.steering_gain * error
        return float(steering), float(speed_values[index])


class RacelineFollower:
    """Non-reactive traffic vehicle that tracks its assigned raceline."""

    def __init__(self, map_name, raceline_file):
        self.conf = expert_configuration()
        raceline_path = racetrack_path(map_name, f"{raceline_file}.csv")
        values = np.loadtxt(raceline_path, delimiter=";", skiprows=1, ndmin=2)
        self.waypoints = np.column_stack((
            values[:, 1],
            values[:, 2],
            np.clip(values[:, 5], 0.0, self.conf.maximum_speed),
            values[:, 3],
            values[:, 0],
        ))
        self.tracker = PurePursuitController(self.conf)

    def reference_trajectory(self, pose_x, pose_y):
        position = np.array([pose_x, pose_y])
        _, _, fraction, segment_index = nearest_point(position, self.waypoints[:, :2])
        start_index = segment_index + int(fraction >= 0.5)
        indices = np.arange(start_index, start_index + self.conf.trajectory_points) % len(self.waypoints)
        trajectory = np.zeros((len(indices), 5))
        trajectory[:, :4] = self.waypoints[indices, :4]
        return trajectory
