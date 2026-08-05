"""PythonRobotics Frenet Optimal Trajectory expert for F1TENTH.

The Frenet Optimal Trajectory implementation in this module is adapted from
Atsushi Sakai et al.'s PythonRobotics project at commit
b38c510e083d69a5755d98d0680bd50f3d9a91fa.

MIT License

Copyright (c) 2016 - now Atsushi Sakai and other PythonRobotics contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from config import load_project_config, merge_config_sections
from utils import nearest_point


LIDAR_FIELD_OF_VIEW = 6.28


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        matrix = np.array(
            [
                [time**3, time**4, time**5],
                [3.0 * time**2, 4.0 * time**3, 5.0 * time**4],
                [6.0 * time, 12.0 * time**2, 20.0 * time**3],
            ]
        )
        vector = np.array(
            [
                xe - self.a0 - self.a1 * time - self.a2 * time**2,
                vxe - self.a1 - 2.0 * self.a2 * time,
                axe - 2.0 * self.a2,
            ]
        )
        self.a3, self.a4, self.a5 = np.linalg.solve(matrix, vector)

    def position(self, time):
        return (
            self.a0
            + self.a1 * time
            + self.a2 * time**2
            + self.a3 * time**3
            + self.a4 * time**4
            + self.a5 * time**5
        )

    def first_derivative(self, time):
        return (
            self.a1
            + 2.0 * self.a2 * time
            + 3.0 * self.a3 * time**2
            + 4.0 * self.a4 * time**3
            + 5.0 * self.a5 * time**4
        )

    def second_derivative(self, time):
        return (
            2.0 * self.a2
            + 6.0 * self.a3 * time
            + 12.0 * self.a4 * time**2
            + 20.0 * self.a5 * time**3
        )

    def third_derivative(self, time):
        return 6.0 * self.a3 + 24.0 * self.a4 * time + 60.0 * self.a5 * time**2


class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        matrix = np.array(
            [
                [3.0 * time**2, 4.0 * time**3],
                [6.0 * time, 12.0 * time**2],
            ]
        )
        vector = np.array(
            [vxe - self.a1 - 2.0 * self.a2 * time, axe - 2.0 * self.a2]
        )
        self.a3, self.a4 = np.linalg.solve(matrix, vector)

    def position(self, time):
        return (
            self.a0
            + self.a1 * time
            + self.a2 * time**2
            + self.a3 * time**3
            + self.a4 * time**4
        )

    def first_derivative(self, time):
        return (
            self.a1
            + 2.0 * self.a2 * time
            + 3.0 * self.a3 * time**2
            + 4.0 * self.a4 * time**3
        )

    def second_derivative(self, time):
        return 2.0 * self.a2 + 6.0 * self.a3 * time + 12.0 * self.a4 * time**2

    def third_derivative(self, time):
        return 6.0 * self.a3 + 24.0 * self.a4 * time


class PeriodicReference:
    """Periodic cubic-spline representation of a closed racing line."""

    def __init__(self, points):
        points = np.asarray(points, dtype=np.float64)
        if np.linalg.norm(points[0] - points[-1]) < 1e-8:
            points = points[:-1]
        self.points = np.vstack((points, points[0]))
        self.segment_lengths = np.linalg.norm(
            np.diff(self.points, axis=0), axis=1
        )
        if np.any(self.segment_lengths <= 0.0):
            raise ValueError("Raceline contains duplicate adjacent points")
        self.s = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.length = float(self.s[-1])
        self.x_spline = CubicSpline(
            self.s, self.points[:, 0], bc_type="periodic"
        )
        self.y_spline = CubicSpline(
            self.s, self.points[:, 1], bc_type="periodic"
        )

    def _wrap(self, distance):
        return float(distance % self.length)

    def position(self, distance):
        distance = self._wrap(distance)
        return (
            float(self.x_spline(distance)),
            float(self.y_spline(distance)),
        )

    def yaw(self, distance):
        distance = self._wrap(distance)
        return math.atan2(
            float(self.y_spline(distance, 1)),
            float(self.x_spline(distance, 1)),
        )

    def curvature(self, distance):
        distance = self._wrap(distance)
        dx = float(self.x_spline(distance, 1))
        dy = float(self.y_spline(distance, 1))
        ddx = float(self.x_spline(distance, 2))
        ddy = float(self.y_spline(distance, 2))
        denominator = max((dx * dx + dy * dy) ** 1.5, 1e-9)
        return (dx * ddy - dy * ddx) / denominator

    def curvature_rate(self, distance):
        epsilon = 0.02
        return (
            self.curvature(distance + epsilon)
            - self.curvature(distance - epsilon)
        ) / (2.0 * epsilon)

    def project(self, x_position, y_position):
        position = np.array([x_position, y_position])
        projection, distance, fraction, index = nearest_point(
            position, self.points
        )
        course_distance = float(
            self.s[index] + fraction * self.segment_lengths[index]
        )
        yaw = self.yaw(course_distance)
        tangent = np.array([math.cos(yaw), math.sin(yaw)])
        cross_product = (
            tangent[0] * (y_position - projection[1])
            - tangent[1] * (x_position - projection[0])
        )
        lateral_distance = (
            math.copysign(float(distance), float(cross_product))
            if distance > 0.0
            else 0.0
        )
        return course_distance, lateral_distance, int(index)


class FrenetPath:
    def __init__(self):
        self.time = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.x = []
        self.y = []
        self.yaw = []
        self.curvature = []
        self.velocity = []
        self.acceleration = []
        self.cost = 0.0


def lidar_scan_to_points(scan, pose):
    scan = np.asarray(scan, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scan) & (scan > 0.0)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    beam_angles = np.linspace(
        -LIDAR_FIELD_OF_VIEW / 2.0,
        LIDAR_FIELD_OF_VIEW / 2.0,
        scan.size,
    )[valid]
    angles = beam_angles + pose[2]
    distances = scan[valid]
    return np.column_stack(
        (
            pose[0] + distances * np.cos(angles),
            pose[1] + distances * np.sin(angles),
        )
    )


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _frenet_to_cartesian(reference, path):
    for index, course_distance in enumerate(path.s):
        reference_x, reference_y = reference.position(course_distance)
        reference_yaw = reference.yaw(course_distance)
        reference_curvature = reference.curvature(course_distance)
        curvature_rate = reference.curvature_rate(course_distance)
        lateral_distance = path.d[index]
        lateral_derivative = path.d_d[index]
        lateral_second_derivative = path.d_dd[index]
        one_minus_curvature_distance = 1.0 - reference_curvature * lateral_distance
        if abs(one_minus_curvature_distance) < 1e-6:
            break

        cosine = math.cos(reference_yaw)
        sine = math.sin(reference_yaw)
        x_position = reference_x - sine * lateral_distance
        y_position = reference_y + cosine * lateral_distance
        heading_offset = math.atan2(
            lateral_derivative, one_minus_curvature_distance
        )
        heading = _normalize_angle(reference_yaw + heading_offset)
        heading_cosine = math.cos(heading_offset)
        tangent = lateral_derivative / one_minus_curvature_distance
        curvature_distance_rate = (
            curvature_rate * lateral_distance
            + reference_curvature * lateral_derivative
        )
        curvature = (
            (
                lateral_second_derivative
                + curvature_distance_rate * tangent
            )
            * heading_cosine**2
            / one_minus_curvature_distance
            + reference_curvature
        ) * heading_cosine / one_minus_curvature_distance
        lateral_time_derivative = lateral_derivative * path.s_d[index]
        velocity = math.sqrt(
            one_minus_curvature_distance**2 * path.s_d[index] ** 2
            + lateral_time_derivative**2
        )
        heading_rate = (
            one_minus_curvature_distance / heading_cosine * curvature
            - reference_curvature
        )
        acceleration = (
            path.s_dd[index]
            * one_minus_curvature_distance
            / heading_cosine
            + path.s_d[index] ** 2
            / heading_cosine
            * (
                lateral_derivative * heading_rate
                - curvature_distance_rate
            )
        )
        path.x.append(x_position)
        path.y.append(y_position)
        path.yaw.append(heading)
        path.curvature.append(curvature)
        path.velocity.append(velocity)
        path.acceleration.append(acceleration)


class TrajectoryTracker:
    """Speed-adaptive lookahead tracker retained after controller validation."""

    def __init__(self, configuration):
        self.min_lookahead = configuration.min_lookahead
        self.max_lookahead = configuration.max_lookahead
        self.lookahead_speed_scale = configuration.lookahead_speed_scale
        self.min_steering_gain = configuration.min_steering_gain
        self.max_steering_gain = configuration.max_steering_gain
        self.steering_speed_scale = configuration.steering_speed_scale
        self.derivative_gain = configuration.steering_derivative_gain
        self.interpolation_points = configuration.interpolation_points
        self.previous_error = 0.0

    def plan(self, pose_x, pose_y, pose_theta, current_speed, trajectory):
        lookahead = (
            current_speed
            * (self.max_lookahead - self.min_lookahead)
            / self.lookahead_speed_scale
            + self.min_lookahead
        )
        steering_gain = (
            self.max_steering_gain
            - current_speed
            * (self.max_steering_gain - self.min_steering_gain)
            / self.steering_speed_scale
        )
        position = np.array([pose_x, pose_y])
        distances = np.linalg.norm(trajectory[:, :2] - position, axis=1)
        segment_end = int(np.argmin(distances))
        if distances[-1] < lookahead:
            segment_end = len(trajectory) - 1
        else:
            while (
                segment_end + 1 < len(trajectory)
                and distances[segment_end] < lookahead
            ):
                segment_end += 1
        segment_start = max(segment_end - 1, 0)
        x_values = np.linspace(
            trajectory[segment_start, 0],
            trajectory[segment_end, 0],
            self.interpolation_points,
        )
        y_values = np.linspace(
            trajectory[segment_start, 1],
            trajectory[segment_end, 1],
            self.interpolation_points,
        )
        speed_values = np.linspace(
            trajectory[segment_start, 2],
            trajectory[segment_end, 2],
            self.interpolation_points,
        )
        interpolated = np.column_stack((x_values, y_values))
        index = int(
            np.argmin(
                np.abs(np.linalg.norm(interpolated - position, axis=1) - lookahead)
            )
        )
        target = interpolated[index]
        actual_lookahead = max(np.linalg.norm(position - target), 1e-6)
        lateral_error = np.dot(
            np.array([math.sin(-pose_theta), math.cos(-pose_theta)]),
            target - position,
        )
        error = 2.0 * lateral_error / actual_lookahead**2
        steering = (
            steering_gain * error
            + self.derivative_gain * (error - self.previous_error)
        )
        self.previous_error = error
        return float(steering), float(speed_values[index])


class FrenetOptimalTrajectoryPlanner:
    def __init__(self, configuration, map_path, raceline_path):
        self.conf = configuration
        self.map_path = map_path
        values = np.loadtxt(
            raceline_path, delimiter=";", skiprows=1, ndmin=2
        )
        self.waypoints = np.column_stack(
            (
                values[:, 1],
                values[:, 2],
                values[:, 5],
                values[:, 3],
                values[:, 0],
            )
        )
        self.reference = PeriodicReference(self.waypoints[:, :2])
        self.reference_speeds = self.waypoints[:, 2]
        self.maximum_speed = configuration.maximum_speed
        self.best_trajectory = None
        self.fallback_count = 0
        self.tracker = TrajectoryTracker(configuration)

    def _terminal_speeds(self, target_speed):
        speeds = np.arange(
            target_speed
            - self.conf.target_speed_step * self.conf.target_speed_samples,
            target_speed
            + self.conf.target_speed_step * self.conf.target_speed_samples,
            self.conf.target_speed_step,
        )
        bounded = []
        for speed in speeds:
            speed = float(np.clip(speed, 0.0, self.maximum_speed))
            if all(abs(speed - existing) > 1e-6 for existing in bounded):
                bounded.append(speed)
        return bounded

    def _target_speed(self, waypoint_index, current_speed):
        horizon_distance = self.conf.speed_lookahead_base
        horizon_distance += max(current_speed, 1.0) * self.conf.max_time
        point_spacing = self.reference.length / len(self.reference_speeds)
        point_count = max(2, int(math.ceil(horizon_distance / point_spacing)))
        indices = (
            np.arange(waypoint_index, waypoint_index + point_count)
            % len(self.reference_speeds)
        )
        return float(
            np.clip(
                np.min(self.reference_speeds[indices]),
                0.0,
                self.maximum_speed,
            )
        )

    def _generate_paths(
        self,
        course_distance,
        course_speed,
        course_acceleration,
        lateral_distance,
        lateral_derivative,
        lateral_second_derivative,
        target_speed,
    ):
        paths = []
        lateral_targets = np.arange(
            -self.conf.road_width,
            self.conf.road_width,
            self.conf.road_step,
        )
        horizons = np.arange(
            self.conf.min_time,
            self.conf.max_time,
            self.conf.time_step,
        )
        for horizon in horizons:
            times = list(np.arange(0.0, horizon, self.conf.time_step))
            for terminal_speed in self._terminal_speeds(target_speed):
                longitudinal = QuarticPolynomial(
                    course_distance,
                    course_speed,
                    course_acceleration,
                    terminal_speed,
                    0.0,
                    horizon,
                )
                base_s = [longitudinal.position(time) for time in times]
                base_s_d = [
                    longitudinal.first_derivative(time) for time in times
                ]
                base_s_dd = [
                    longitudinal.second_derivative(time) for time in times
                ]
                base_s_ddd = [
                    longitudinal.third_derivative(time) for time in times
                ]
                for lateral_target in lateral_targets:
                    path = FrenetPath()
                    path.time = times
                    path.s = base_s
                    path.s_d = base_s_d
                    path.s_dd = base_s_dd
                    path.s_ddd = base_s_ddd
                    lateral = QuinticPolynomial(
                        lateral_distance,
                        lateral_derivative * base_s_d[0],
                        lateral_second_derivative * base_s_d[0] ** 2
                        + lateral_derivative * base_s_dd[0],
                        lateral_target,
                        0.0,
                        0.0,
                        horizon,
                    )
                    for index, time in enumerate(times):
                        time_first = lateral.first_derivative(time)
                        time_second = lateral.second_derivative(time)
                        inverse_speed = 1.0 / (base_s_d[index] + 1e-6) + 1e-6
                        distance_first = time_first * inverse_speed
                        path.d.append(lateral.position(time))
                        path.d_d.append(distance_first)
                        path.d_dd.append(
                            (
                                time_second
                                - distance_first * base_s_dd[index]
                            )
                            * inverse_speed**2
                        )
                        path.d_ddd.append(lateral.third_derivative(time))

                    lateral_jerk = sum(np.square(path.d_ddd))
                    longitudinal_jerk = sum(np.square(path.s_ddd))
                    lateral_cost = (
                        self.conf.jerk_cost * lateral_jerk
                        + self.conf.time_cost * horizon
                        + self.conf.lateral_offset_cost * path.d[-1] ** 2
                    )
                    speed_error = (target_speed - path.s_d[-1]) ** 2
                    longitudinal_cost = (
                        self.conf.jerk_cost * longitudinal_jerk
                        + self.conf.time_cost * horizon
                        + self.conf.speed_cost * speed_error
                    )
                    path.cost = (
                        self.conf.lateral_cost * lateral_cost
                        + self.conf.longitudinal_cost * longitudinal_cost
                    )
                    _frenet_to_cartesian(self.reference, path)
                    paths.append(path)
        return paths

    def _collision_free(self, path, obstacle_tree):
        if obstacle_tree is None or not path.x:
            return True
        distances, _ = obstacle_tree.query(
            np.column_stack((path.x, path.y)), k=1
        )
        return bool(np.all(distances > self.conf.clearance_radius))

    def _best_path(self, paths, obstacle_tree):
        best_path = None
        best_cost = math.inf
        for path in paths:
            if not path.x:
                continue
            if any(speed > self.maximum_speed for speed in path.velocity):
                continue
            if any(
                abs(acceleration) > self.conf.max_acceleration
                for acceleration in path.acceleration
            ):
                continue
            if any(
                abs(curvature) > self.conf.max_curvature
                for curvature in path.curvature
            ):
                continue
            if not self._collision_free(path, obstacle_tree):
                continue
            if path.cost <= best_cost:
                best_cost = path.cost
                best_path = path
        return best_path

    def _fallback(self, course_distance, lateral_distance, speed):
        self.fallback_count += 1
        distance = max(
            self.conf.fallback_min_distance,
            speed * self.conf.fallback_time,
        )
        distances = np.linspace(0.0, distance, self.conf.trajectory_points)
        target_speed = max(0.0, speed - self.conf.fallback_speed_reduction)
        trajectory = np.zeros((len(distances), 5))
        trajectory[:, 2] = target_speed
        for index, offset in enumerate(distances):
            path_distance = course_distance + offset
            reference_x, reference_y = self.reference.position(path_distance)
            reference_yaw = self.reference.yaw(path_distance)
            trajectory[index, 0] = (
                reference_x - math.sin(reference_yaw) * lateral_distance
            )
            trajectory[index, 1] = (
                reference_y + math.cos(reference_yaw) * lateral_distance
            )
            trajectory[index, 3] = reference_yaw
            trajectory[index, 4] = self.reference.curvature(path_distance)
        return trajectory

    def plan(self, pose_x, pose_y, pose_theta, lidar_scan, velocity):
        course_distance, lateral_distance, waypoint_index = self.reference.project(
            pose_x, pose_y
        )
        reference_yaw = self.reference.yaw(course_distance)
        reference_curvature = self.reference.curvature(course_distance)
        heading_error = _normalize_angle(pose_theta - reference_yaw)
        denominator = max(
            1.0 - reference_curvature * lateral_distance, 0.2
        )
        course_speed = max(
            velocity * math.cos(heading_error) / denominator, 0.05
        )
        lateral_derivative = float(
            np.clip(denominator * math.tan(heading_error), -1.5, 1.5)
        )
        target_speed = self._target_speed(waypoint_index, velocity)
        paths = self._generate_paths(
            course_distance,
            course_speed,
            0.0,
            lateral_distance,
            lateral_derivative,
            0.0,
            target_speed,
        )

        obstacle_tree = None
        obstacles = lidar_scan_to_points(
            lidar_scan,
            np.array([pose_x, pose_y, pose_theta]),
        )
        if len(obstacles):
            obstacle_tree = cKDTree(obstacles)
        path = self._best_path(paths, obstacle_tree)
        if path is None or len(path.x) < 2:
            self.best_trajectory = self._fallback(
                course_distance, lateral_distance, velocity
            )
            return self.best_trajectory

        trajectory = np.zeros((len(path.x), 5))
        trajectory[:, 0] = path.x
        trajectory[:, 1] = path.y
        trajectory[:, 2] = np.clip(
            path.velocity, 0.0, self.maximum_speed
        )
        trajectory[:, 3] = path.yaw
        trajectory[:, 4] = path.curvature
        self.best_trajectory = trajectory
        return trajectory


class RacelineFollower:
    """Non-reactive traffic vehicle that tracks its assigned raceline."""

    def __init__(self, configuration, map_path, raceline_path):
        self.conf = configuration
        self.map_path = map_path
        values = np.loadtxt(
            raceline_path, delimiter=";", skiprows=1, ndmin=2
        )
        self.waypoints = np.column_stack(
            (
                values[:, 1],
                values[:, 2],
                np.clip(values[:, 5], 0.0, configuration.maximum_speed),
                values[:, 3],
                values[:, 0],
            )
        )
        self.best_trajectory = None
        self.tracker = TrajectoryTracker(configuration)

    def plan(self, pose_x, pose_y, pose_theta, lidar_scan, velocity):
        del pose_theta, lidar_scan, velocity
        position = np.array([pose_x, pose_y])
        _, _, fraction, segment_index = nearest_point(
            position, self.waypoints[:, :2]
        )
        start_index = segment_index + int(fraction >= 0.5)
        indices = (
            np.arange(
                start_index,
                start_index + self.conf.trajectory_points,
            )
            % len(self.waypoints)
        )
        trajectory = np.zeros((len(indices), 5))
        trajectory[:, :4] = self.waypoints[indices, :4]
        self.best_trajectory = trajectory
        return trajectory


def _expert_paths(map_name, raceline_file):
    map_directory = Path("f1tenth_racetracks") / map_name
    return (
        map_directory,
        map_directory / f"{map_name}_map",
        map_directory / f"{raceline_file}.csv",
    )


def create_expert_planner(map_name, raceline_file):
    project = load_project_config()
    map_directory, map_path, raceline_path = _expert_paths(
        map_name, raceline_file
    )
    planner = FrenetOptimalTrajectoryPlanner(
        merge_config_sections(project.expert, project.vehicle),
        str(map_path),
        raceline_path,
    )
    return planner, str(map_directory)


def create_opponent(map_name, raceline_file):
    project = load_project_config()
    map_directory, map_path, raceline_path = _expert_paths(
        map_name, raceline_file
    )
    opponent = RacelineFollower(
        merge_config_sections(project.expert, project.vehicle),
        str(map_path),
        raceline_path,
    )
    return opponent, str(map_directory)
