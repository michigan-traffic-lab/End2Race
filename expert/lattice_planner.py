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

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from expert.utils import load_expert_config, nearest_point
from f1tenth_sim.utils import load_racetrack_config, racetrack_path


LIDAR_FIELD_OF_VIEW = 6.28


def _expert_configuration():
    return SimpleNamespace(
        **vars(load_expert_config().expert),
        **vars(load_racetrack_config().vehicle),
    )


class QuarticLateralPolynomial:
    """Lateral motion with free velocity and zero acceleration at the end."""

    def __init__(self, xs, vxs, axs, xe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        matrix = np.array([[time**3, time**4], [6.0 * time, 12.0 * time**2]])
        vector = np.array([
            xe - self.a0 - self.a1 * time - self.a2 * time**2,
            axe - 2.0 * self.a2,
        ])
        self.a3, self.a4 = np.linalg.solve(matrix, vector)

    def position(self, time):
        return self.a0 + self.a1 * time + self.a2 * time**2 + self.a3 * time**3 + self.a4 * time**4

    def first_derivative(self, time):
        return self.a1 + 2.0 * self.a2 * time + 3.0 * self.a3 * time**2 + 4.0 * self.a4 * time**3

    def second_derivative(self, time):
        return 2.0 * self.a2 + 6.0 * self.a3 * time + 12.0 * self.a4 * time**2


class PeriodicReference:
    """Periodic cubic-spline representation of a closed racing line."""

    def __init__(self, points):
        points = np.asarray(points, dtype=np.float64)
        if np.linalg.norm(points[0] - points[-1]) < 1e-8:
            points = points[:-1]
        self.points = np.vstack((points, points[0]))
        self.segment_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        if np.any(self.segment_lengths <= 0.0):
            raise ValueError("Raceline contains duplicate adjacent points")
        self.s = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.length = float(self.s[-1])
        self.x_spline = CubicSpline(self.s, self.points[:, 0], bc_type="periodic")
        self.y_spline = CubicSpline(self.s, self.points[:, 1], bc_type="periodic")

    def _wrap(self, distance):
        return float(distance % self.length)

    def position(self, distance):
        distance = self._wrap(distance)
        return float(self.x_spline(distance)), float(self.y_spline(distance))

    def yaw(self, distance):
        distance = self._wrap(distance)
        return math.atan2(float(self.y_spline(distance, 1)), float(self.x_spline(distance, 1)))

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
        return (self.curvature(distance + epsilon) - self.curvature(distance - epsilon)) / (2.0 * epsilon)

    def project(self, x_position, y_position):
        position = np.array([x_position, y_position])
        projection, distance, fraction, index = nearest_point(position, self.points)
        course_distance = float(self.s[index] + fraction * self.segment_lengths[index])
        yaw = self.yaw(course_distance)
        tangent = np.array([math.cos(yaw), math.sin(yaw)])
        cross_product = (
            tangent[0] * (y_position - projection[1])
            - tangent[1] * (x_position - projection[0])
        )
        lateral_distance = (
            math.copysign(float(distance), float(cross_product)) if distance > 0.0 else 0.0
        )
        return course_distance, lateral_distance, int(index)


class FrenetPath:
    def __init__(self):
        self.time = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.x = []
        self.y = []
        self.yaw = []
        self.curvature = []
        self.velocity = []
        self.acceleration = []
        self.target_velocity = None


_FOT_CANDIDATE_WORKER = None


def _initialize_fot_candidate_worker(configuration, map_path, raceline_path):
    global _FOT_CANDIDATE_WORKER
    _FOT_CANDIDATE_WORKER = FrenetOptimalTrajectoryPlanner(configuration, map_path, raceline_path)


def _generate_lateral_paths_in_worker(arguments):
    return _FOT_CANDIDATE_WORKER._generate_lateral_paths(*arguments)


def lidar_scan_to_points(scan, pose):
    scan = np.asarray(scan, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scan) & (scan > 0.0)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    beam_angles = np.linspace(-LIDAR_FIELD_OF_VIEW / 2.0, LIDAR_FIELD_OF_VIEW / 2.0, scan.size)[valid]
    angles = beam_angles + pose[2]
    distances = scan[valid]
    return np.column_stack((
        pose[0] + distances * np.cos(angles),
        pose[1] + distances * np.sin(angles),
    ))


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
        heading_offset = math.atan2(lateral_derivative, one_minus_curvature_distance)
        heading = _normalize_angle(reference_yaw + heading_offset)
        heading_cosine = math.cos(heading_offset)
        if abs(heading_cosine) < 1e-6:
            break
        tangent = lateral_derivative / one_minus_curvature_distance
        curvature_distance_rate = (
            curvature_rate * lateral_distance + reference_curvature * lateral_derivative
        )
        curvature = (
            (lateral_second_derivative + curvature_distance_rate * tangent)
            * heading_cosine**2
            / one_minus_curvature_distance
            + reference_curvature
        ) * heading_cosine / one_minus_curvature_distance
        path.x.append(x_position)
        path.y.append(y_position)
        path.yaw.append(heading)
        path.curvature.append(curvature)
        lateral_time_derivative = lateral_derivative * path.s_d[index]
        velocity = math.sqrt(
            one_minus_curvature_distance**2 * path.s_d[index] ** 2 + lateral_time_derivative**2
        )
        heading_rate = one_minus_curvature_distance / heading_cosine * curvature - reference_curvature
        acceleration = (
            path.s_dd[index] * one_minus_curvature_distance / heading_cosine
            + path.s_d[index] ** 2
            / heading_cosine
            * (lateral_derivative * heading_rate - curvature_distance_rate)
        )
        path.velocity.append(velocity)
        path.acceleration.append(acceleration)


class TrajectoryTracker:
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


class FrenetOptimalTrajectoryPlanner:
    def __init__(self, configuration, map_path, raceline_path):
        self.conf = configuration
        self.map_path = map_path
        self.raceline_path = Path(raceline_path)
        values = np.loadtxt(self.raceline_path, delimiter=";", skiprows=1, ndmin=2)
        self.waypoints = np.column_stack(
            (values[:, 1], values[:, 2], values[:, 5], values[:, 3], values[:, 0])
        )
        self.reference = PeriodicReference(self.waypoints[:, :2])
        self.maximum_speed = configuration.maximum_speed
        self.best_trajectory = None
        self.tracker = TrajectoryTracker(configuration)
        self.parallel_workers = configuration.parallel_workers
        self._candidate_pool = None

    def _terminal_speed_limits(self, physical_speed):
        minimum_speed = max(
            physical_speed - self.conf.horizon * self.conf.max_acceleration,
            self.conf.minimum_terminal_speed,
        )
        maximum_speed = min(
            physical_speed + self.conf.horizon * self.conf.max_acceleration,
            self.maximum_speed,
        )
        return minimum_speed, maximum_speed

    def _terminal_speeds(self, physical_speed, minimum_speed=None, maximum_speed=None):
        if minimum_speed is None or maximum_speed is None:
            minimum_speed, maximum_speed = self._terminal_speed_limits(physical_speed)
        reference_speed = float(np.clip(physical_speed, minimum_speed, maximum_speed))
        return [
            minimum_speed,
            0.5 * (minimum_speed + reference_speed),
            reference_speed,
            0.5 * (reference_speed + maximum_speed),
            maximum_speed,
        ]

    def _lateral_targets(self):
        if self.conf.road_width <= 0.0 or self.conf.road_step <= 0.0:
            raise ValueError("expert.road_width and expert.road_step must be positive")
        step_count = math.floor(self.conf.road_width / self.conf.road_step + 1e-9)
        if step_count < 1:
            raise ValueError("expert.road_step must not exceed expert.road_width")
        return np.arange(-step_count, step_count + 1) * self.conf.road_step

    def _sample_times(self):
        interval_count = round(self.conf.horizon / self.conf.time_step)
        if not math.isclose(
            interval_count * self.conf.time_step, self.conf.horizon, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("expert.horizon must be divisible by expert.time_step")
        return np.linspace(0.0, self.conf.horizon, interval_count + 1)

    def _course_terminal_speed(
        self, course_distance, course_speed, lateral_target,
        physical_terminal_speed, lateral_terminal_velocity,
    ):
        tangential_speed_squared = physical_terminal_speed**2 - lateral_terminal_velocity**2
        if tangential_speed_squared <= 0.0:
            return None
        physical_tangential_speed = math.sqrt(tangential_speed_squared)
        terminal_course_speed = physical_terminal_speed
        # The terminal course speed depends on the curvature at the endpoint it determines, so approximate it with a bounded fixed-point iteration
        for _ in range(8):
            course_acceleration = (terminal_course_speed - course_speed) / self.conf.horizon
            terminal_distance = (
                course_distance
                + course_speed * self.conf.horizon
                + 0.5 * course_acceleration * self.conf.horizon**2
            )
            denominator = 1.0 - self.reference.curvature(terminal_distance) * lateral_target
            if denominator <= 0.2:
                return None
            updated_speed = physical_tangential_speed / denominator
            if math.isclose(updated_speed, terminal_course_speed, rel_tol=0.0, abs_tol=1e-8):
                return updated_speed
            terminal_course_speed = updated_speed
        return terminal_course_speed

    def _generate_candidate(
        self, course_distance, course_speed, lateral_distance, lateral_time_velocity,
        lateral_target, physical_terminal_speed, sample_times, require_dynamic_feasibility=True,
    ):
        lateral = QuarticLateralPolynomial(
            lateral_distance, lateral_time_velocity, 0.0, lateral_target, 0.0, self.conf.horizon
        )
        terminal_course_speed = self._course_terminal_speed(
            course_distance,
            course_speed,
            lateral_target,
            physical_terminal_speed,
            lateral.first_derivative(self.conf.horizon),
        )
        if terminal_course_speed is None:
            return None

        course_acceleration = (terminal_course_speed - course_speed) / self.conf.horizon
        path = FrenetPath()
        path.target_velocity = physical_terminal_speed
        path.time = sample_times.copy()
        for time in path.time:
            course_speed_at_time = course_speed + course_acceleration * time
            if course_speed_at_time <= 1e-6:
                return None
            lateral_time_derivative = lateral.first_derivative(time)
            lateral_time_second = lateral.second_derivative(time)
            lateral_derivative = lateral_time_derivative / course_speed_at_time
            lateral_second = (
                lateral_time_second - lateral_derivative * course_acceleration
            ) / course_speed_at_time**2
            path.s.append(
                course_distance + course_speed * time + 0.5 * course_acceleration * time**2
            )
            path.s_d.append(course_speed_at_time)
            path.s_dd.append(course_acceleration)
            path.d.append(lateral.position(time))
            path.d_d.append(lateral_derivative)
            path.d_dd.append(lateral_second)

        _frenet_to_cartesian(self.reference, path)
        if len(path.x) != len(path.time):
            return None
        if require_dynamic_feasibility and not self._is_dynamically_feasible(
            path, physical_terminal_speed
        ):
            return None
        return path

    @staticmethod
    def _feasible_boundary(infeasible_speed, feasible_speed, candidate_for_speed):
        """Find the feasibility boundary from its feasible side."""
        for _ in range(40):
            midpoint = 0.5 * (infeasible_speed + feasible_speed)
            if abs(infeasible_speed - feasible_speed) <= 1e-6:
                break
            if candidate_for_speed(midpoint) is None:
                infeasible_speed = midpoint
            else:
                feasible_speed = midpoint
        return feasible_speed

    def _feasible_terminal_speeds(
        self, course_distance, course_speed, lateral_distance,
        lateral_time_velocity, lateral_target, physical_speed, sample_times,
    ):
        nominal_minimum, nominal_maximum = self._terminal_speed_limits(physical_speed)
        candidate_cache = {}

        def candidate_for_speed(terminal_speed):
            terminal_speed = float(terminal_speed)
            if terminal_speed not in candidate_cache:
                candidate_cache[terminal_speed] = self._generate_candidate(
                    course_distance,
                    course_speed,
                    lateral_distance,
                    lateral_time_velocity,
                    lateral_target,
                    terminal_speed,
                    sample_times,
                )
            return candidate_cache[terminal_speed]

        nominal_speeds = self._terminal_speeds(physical_speed, nominal_minimum, nominal_maximum)
        feasible_seeds = [
            speed for speed in nominal_speeds if candidate_for_speed(speed) is not None
        ]
        if not feasible_seeds:
            return [], candidate_cache

        if candidate_for_speed(nominal_minimum) is not None:
            feasible_minimum = nominal_minimum
        else:
            feasible_minimum = self._feasible_boundary(
                nominal_minimum, min(feasible_seeds), candidate_for_speed
            )

        if candidate_for_speed(nominal_maximum) is not None:
            feasible_maximum = nominal_maximum
        else:
            feasible_maximum = self._feasible_boundary(
                nominal_maximum, max(feasible_seeds), candidate_for_speed
            )

        terminal_speeds = self._terminal_speeds(physical_speed, feasible_minimum, feasible_maximum)
        return list(dict.fromkeys(terminal_speeds)), candidate_cache

    def _is_dynamically_feasible(self, path, physical_terminal_speed):
        velocity = np.asarray(path.velocity)
        acceleration = np.asarray(path.acceleration)
        curvature = np.asarray(path.curvature)
        # The first sample is the t=0 trajectory anchor; enforce dynamic limits only on future samples
        future = slice(1, None)
        return not (
            np.any(~np.isfinite(velocity))
            or np.any(~np.isfinite(acceleration))
            or np.any(~np.isfinite(curvature))
            or np.any(velocity[future] < -1e-9)
            or np.any(velocity[future] > self.maximum_speed + 1e-9)
            or np.any(np.abs(acceleration[future]) > self.conf.max_acceleration + 1e-9)
            or np.any(np.abs(curvature[future]) > self.conf.max_curvature + 1e-9)
            or np.any(
                velocity[future] ** 2 * np.abs(curvature[future])
                > self.conf.max_lateral_acceleration + 1e-9
            )
            or not math.isclose(
                velocity[-1], physical_terminal_speed, rel_tol=0.0, abs_tol=1e-5
            )
        )

    def _generate_paths(
        self, course_distance, lateral_distance, heading_error, physical_speed,
        require_dynamic_feasibility=True,
    ):
        reference_curvature = self.reference.curvature(course_distance)
        denominator = max(1.0 - reference_curvature * lateral_distance, 0.2)
        course_speed = max(physical_speed * math.cos(heading_error) / denominator, 0.05)
        lateral_time_velocity = physical_speed * math.sin(heading_error)
        sample_times = self._sample_times().tolist()
        lateral_arguments = [
            (
                course_distance,
                course_speed,
                lateral_distance,
                lateral_time_velocity,
                float(lateral_target),
                physical_speed,
                sample_times,
                require_dynamic_feasibility,
            )
            for lateral_target in self._lateral_targets()
        ]
        # Workers only ever run _generate_lateral_paths, which never reaches this branch
        if self.parallel_workers > 1 and len(lateral_arguments) > 1:
            if self._candidate_pool is None:
                self._candidate_pool = ProcessPoolExecutor(
                    max_workers=min(self.parallel_workers, len(lateral_arguments)),
                    mp_context=mp.get_context("fork"),
                    initializer=_initialize_fot_candidate_worker,
                    initargs=(self.conf, self.map_path, str(self.raceline_path)),
                )
            lateral_path_groups = self._candidate_pool.map(
                _generate_lateral_paths_in_worker, lateral_arguments
            )
            return [path for lateral_paths in lateral_path_groups for path in lateral_paths]

        paths = []
        for arguments in lateral_arguments:
            paths.extend(self._generate_lateral_paths(*arguments))
        return paths

    def _generate_lateral_paths(
        self, course_distance, course_speed, lateral_distance, lateral_time_velocity,
        lateral_target, physical_speed, sample_times, require_dynamic_feasibility,
    ):
        if require_dynamic_feasibility:
            terminal_speeds, candidate_cache = self._feasible_terminal_speeds(
                course_distance,
                course_speed,
                lateral_distance,
                lateral_time_velocity,
                lateral_target,
                physical_speed,
                sample_times,
            )
        else:
            terminal_speeds = self._terminal_speeds(physical_speed)
            candidate_cache = {}

        paths = []
        for terminal_speed in terminal_speeds:
            terminal_speed = float(terminal_speed)
            if terminal_speed in candidate_cache:
                path = candidate_cache[terminal_speed]
            else:
                path = self._generate_candidate(
                    course_distance,
                    course_speed,
                    lateral_distance,
                    lateral_time_velocity,
                    lateral_target,
                    terminal_speed,
                    sample_times,
                    require_dynamic_feasibility,
                )
            if path is not None:
                paths.append(path)
        return paths

    def close(self):
        if self._candidate_pool is not None:
            self._candidate_pool.shutdown()
            self._candidate_pool = None

    def _dynamic_violation(self, path):
        velocity = np.asarray(path.velocity)
        acceleration = np.asarray(path.acceleration)
        curvature = np.asarray(path.curvature)
        if (
            np.any(~np.isfinite(velocity))
            or np.any(~np.isfinite(acceleration))
            or np.any(~np.isfinite(curvature))
        ):
            return math.inf

        future_velocity = velocity[1:]
        future_acceleration = acceleration[1:]
        future_curvature = curvature[1:]
        terminal_error = abs(velocity[-1] - path.target_velocity)
        # Normalize each dynamic-constraint violation and rank the path by the worst one
        violations = (
            max(0.0, -float(np.min(future_velocity))) / self.maximum_speed,
            max(0.0, float(np.max(future_velocity)) / self.maximum_speed - 1.0),
            max(
                0.0,
                float(np.max(np.abs(future_acceleration))) / self.conf.max_acceleration - 1.0,
            ),
            max(0.0, float(np.max(np.abs(future_curvature))) / self.conf.max_curvature - 1.0),
            max(
                0.0,
                float(np.max(future_velocity**2 * np.abs(future_curvature)))
                / self.conf.max_lateral_acceleration
                - 1.0,
            ),
            max(0.0, terminal_error - 1e-5) / self.maximum_speed,
        )
        return max(violations)

    def _best_effort_path(self, paths):
        return min(paths, key=self._dynamic_violation) if paths else None

    def _collision_distances(self, path, obstacle_tree):
        if obstacle_tree is None:
            return np.full(len(path.x) - 1, math.inf)
        future_points = np.column_stack((np.asarray(path.x)[1:], np.asarray(path.y)[1:]))
        distances, _ = obstacle_tree.query(future_points, k=1)
        return distances

    def _best_path(self, paths, obstacle_tree):
        best_path = None
        best_cost = math.inf
        for path in paths:
            future_velocity = np.clip(np.asarray(path.velocity)[1:], 0.0, self.maximum_speed)
            velocity_costs = 1.0 - future_velocity / self.maximum_speed
            collision_distances = self._collision_distances(path, obstacle_tree)
            normalized_deficits = np.maximum(
                0.0, 1.0 - collision_distances / self.conf.soft_collision_clearance
            )
            collision_costs = normalized_deficits**self.conf.collision_cost_power
            selection_cost = float(
                np.mean(velocity_costs)
                + self.conf.collision_cost_weight * np.max(collision_costs)
            )
            if selection_cost < best_cost:
                best_cost = selection_cost
                best_path = path
        return best_path

    def plan(self, pose_x, pose_y, pose_theta, lidar_scan, velocity):
        course_distance, lateral_distance, _ = self.reference.project(pose_x, pose_y)
        reference_yaw = self.reference.yaw(course_distance)
        heading_error = _normalize_angle(pose_theta - reference_yaw)
        paths = self._generate_paths(course_distance, lateral_distance, heading_error, velocity)

        obstacle_tree = None
        obstacles = lidar_scan_to_points(lidar_scan, np.array([pose_x, pose_y, pose_theta]))
        if len(obstacles):
            obstacle_tree = cKDTree(obstacles)
        path = self._best_path(paths, obstacle_tree)
        if path is None:
            # If no dynamically feasible path exists, choose the smallest dynamic violation
            best_effort_paths = self._generate_paths(
                course_distance,
                lateral_distance,
                heading_error,
                velocity,
                require_dynamic_feasibility=False,
            )
            path = self._best_effort_path(best_effort_paths)
        if path is None:
            raise RuntimeError("FOT produced no usable trajectory")

        trajectory = np.column_stack((path.x, path.y, path.velocity, path.yaw, path.curvature))
        self.best_trajectory = trajectory
        return trajectory


class RacelineFollower:
    """Non-reactive traffic vehicle that tracks its assigned raceline."""

    def __init__(self, configuration, raceline_path):
        self.conf = configuration
        values = np.loadtxt(raceline_path, delimiter=";", skiprows=1, ndmin=2)
        self.waypoints = np.column_stack((
            values[:, 1],
            values[:, 2],
            np.clip(values[:, 5], 0.0, configuration.maximum_speed),
            values[:, 3],
            values[:, 0],
        ))
        self.best_trajectory = None
        self.tracker = TrajectoryTracker(configuration)

    def plan(self, pose_x, pose_y, pose_theta, lidar_scan, velocity):
        del pose_theta, lidar_scan, velocity
        position = np.array([pose_x, pose_y])
        _, _, fraction, segment_index = nearest_point(position, self.waypoints[:, :2])
        start_index = segment_index + int(fraction >= 0.5)
        indices = np.arange(start_index, start_index + self.conf.trajectory_points) % len(self.waypoints)
        trajectory = np.zeros((len(indices), 5))
        trajectory[:, :4] = self.waypoints[indices, :4]
        self.best_trajectory = trajectory
        return trajectory


def create_expert_planner(map_name, raceline_file):
    map_path = racetrack_path(map_name, f"{map_name}_map")
    raceline_path = racetrack_path(map_name, f"{raceline_file}.csv")
    return FrenetOptimalTrajectoryPlanner(
        _expert_configuration(), str(map_path), raceline_path
    )


def create_opponent(map_name, raceline_file):
    raceline_path = racetrack_path(map_name, f"{raceline_file}.csv")
    return RacelineFollower(_expert_configuration(), raceline_path)
