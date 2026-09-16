"""Reactive FTG. No map, waypoint, pose, or learned-policy inputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FTGConfig:
    # The bundled simulator uses 1440 beams spanning 6.28 radians.
    lidar_fov: float = 6.28
    forward_fov: float = 2.8
    max_range: float = 21.668276301540864
    smoothing: int = 5
    bubble_radius: float = 0.3521825071214156
    disparity_threshold: float = 0.5
    vehicle_margin: float = 0.2724764267006296
    target_window: int = 61
    lookahead: float = 1.340642097085112
    lookahead_speed: float = 0.12690306426137982
    steering_smoothing: float = 0.7637497934492807
    lateral_accel: float = 9.276001332072708
    braking_accel: float = 7.03846783492933
    max_speed: float = 7.5
    min_speed: float = 2.0
    wheelbase: float = 0.3302
    steering_limit: float = 0.4189


class FollowTheGapController:
    """Return (steering radians, desired speed m/s) at each control tick.

    Accept either a raw one-dimensional scan or the simulator observation.
    Only scans and measured longitudinal speed are read from observations.
    Call reset() for each episode; use one instance per vehicle.
    """

    def __init__(self, configuration=None):
        self.config = configuration or FTGConfig()
        self.reset()

    def reset(self):
        self.previous_steering = 0.0

    def plan(self, scan, current_speed=0.0, agent_index=0):
        if isinstance(scan, dict):
            current_speed = float(scan["linear_vels_x"][agent_index])
            scan = scan["scans"][agent_index]
        c = self.config
        raw = np.asarray(scan, dtype=float)
        angles = np.linspace(-c.lidar_fov / 2, c.lidar_fov / 2, len(raw))
        mask = np.abs(angles) <= c.forward_fov / 2
        angles = angles[mask]
        ranges = np.clip(np.nan_to_num(raw[mask], nan=0.0, posinf=c.max_range, neginf=0.0), 0, c.max_range)
        if not np.any(ranges > c.bubble_radius):
            self.reset()
            return 0.0, 0.0
        window = c.smoothing
        ranges = np.convolve(np.pad(ranges, (window // 2, window // 2), mode="edge"), np.ones(window) / window, mode="valid")
        free = ranges.copy()
        increment = angles[1] - angles[0]
        closest = int(np.argmin(ranges))
        radius = np.arcsin(min(1.0, c.bubble_radius / max(ranges[closest], 1e-6)))
        free[np.abs(angles - angles[closest]) <= radius] = 0
        # Extend depth discontinuities by the car's half-width plus clearance.
        for index in np.flatnonzero(np.abs(np.diff(ranges)) > c.disparity_threshold):
            near = index if ranges[index] < ranges[index + 1] else index + 1
            count = int(np.ceil(np.arcsin(min(1.0, c.vehicle_margin / max(ranges[near], 1e-6))) / increment))
            a, b = (near, min(len(free), near + count + 1)) if near == index else (max(0, near - count), near + 1)
            free[a:b] = np.minimum(free[a:b], ranges[near])
        valid = free > c.bubble_radius
        edges = np.diff(np.r_[False, valid, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if not len(starts):
            self.reset()
            return 0.0, 0.0
        gap = int(np.argmax(ends - starts))
        start, end = starts[gap], ends[gap]
        width = min(c.target_window, end - start)
        score = np.convolve(free[start:end], np.ones(width) / width, mode="valid")
        target = start + int(np.argmax(score)) + (width - 1) / 2
        angle = float(np.interp(target, np.arange(len(angles)), angles))
        lookahead = c.lookahead + c.lookahead_speed * max(0.0, current_speed)
        curvature = 2 * np.sin(angle) / lookahead
        steering = float(np.arctan(c.wheelbase * curvature))
        steering = c.steering_smoothing * steering + (1 - c.steering_smoothing) * self.previous_steering
        steering = float(np.clip(steering, -c.steering_limit, c.steering_limit))
        self.previous_steering = steering
        turn_speed = np.sqrt(c.lateral_accel / max(abs(curvature), 1e-4))
        front = float(np.min(ranges[np.abs(angles) < 0.12]))
        stopping_speed = np.sqrt(2 * c.braking_accel * max(0.0, front - c.bubble_radius))
        speed = min(c.max_speed, max(c.min_speed, min(turn_speed, stopping_speed)))
        return steering, float(speed)
