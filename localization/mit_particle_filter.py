"""Standalone port of the MIT RACECAR Monte Carlo localization filter.

The algorithm and default parameters follow the official ROS1 package at
https://github.com/mit-racecar/particle_filter (commit 95613c6).  ROS message
handling and RangeLibc are replaced by NumPy inputs and the distance-transform
ray marcher already used by F1TENTH Gym so the filter runs on Python 3.11.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt

from config import load_config

CONFIG_KEYS = {
    'scan_beams', 'scan_fov', 'seed', 'max_particles', 'angle_step',
    'squash_factor', 'max_range', 'z_short', 'z_max', 'z_rand', 'z_hit',
    'sigma_hit', 'motion_dispersion_x', 'motion_dispersion_y',
    'motion_dispersion_theta', 'init_position_std', 'init_theta_std',
    'ray_epsilon',
}


@njit(cache=True)
def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def _ray_march_many(
    poses: np.ndarray,
    relative_angles: np.ndarray,
    origin_x: float,
    origin_y: float,
    origin_cos: float,
    origin_sin: float,
    resolution: float,
    distance_transform: np.ndarray,
    max_range: float,
    epsilon: float,
) -> np.ndarray:
    """Cast all particle/beam queries against a distance-transform map."""
    particle_count = poses.shape[0]
    ray_count = relative_angles.shape[0]
    height, width = distance_transform.shape
    ranges = np.empty((particle_count, ray_count), dtype=np.float32)

    for particle_index in range(particle_count):
        start_x = poses[particle_index, 0]
        start_y = poses[particle_index, 1]
        heading = poses[particle_index, 2]

        for ray_index in range(ray_count):
            ray_heading = heading + relative_angles[ray_index]
            ray_cos = np.cos(ray_heading)
            ray_sin = np.sin(ray_heading)
            x = start_x
            y = start_y
            travelled = 0.0

            while travelled < max_range:
                translated_x = x - origin_x
                translated_y = y - origin_y
                map_x = translated_x * origin_cos + translated_y * origin_sin
                map_y = -translated_x * origin_sin + translated_y * origin_cos
                column = int(map_x / resolution)
                row = int(map_y / resolution)

                if row < 0 or row >= height or column < 0 or column >= width:
                    travelled = max_range
                    break

                clearance = distance_transform[row, column]
                if clearance <= epsilon:
                    break

                step = clearance
                if travelled + step >= max_range:
                    travelled = max_range
                    break

                x += step * ray_cos
                y += step * ray_sin
                travelled += step

            ranges[particle_index, ray_index] = travelled

    return ranges


class MITParticleFilter:
    """Monte Carlo localization using odometry increments and LiDAR scans."""

    def __init__(
        self,
        map_path: str,
    ) -> None:
        self.config = load_config('localization/config.yaml', CONFIG_KEYS)
        self.rng = np.random.default_rng(self.config.seed)
        self.scan_beams = int(self.config.scan_beams)
        self.scan_fov = float(self.config.scan_fov)
        self.particle_indices = np.arange(self.config.max_particles)
        self.particles = np.zeros((self.config.max_particles, 3), dtype=np.float64)
        self.weights = np.full(
            self.config.max_particles,
            1.0 / self.config.max_particles,
            dtype=np.float64,
        )
        self.last_odometry_pose: Optional[np.ndarray] = None
        self.inferred_pose: Optional[np.ndarray] = None

        self._load_map(map_path)
        all_angles = np.linspace(-self.scan_fov / 2.0, self.scan_fov / 2.0, self.scan_beams)
        self.downsample_indices = np.arange(0, self.scan_beams, self.config.angle_step)
        self.downsampled_angles = np.ascontiguousarray(
            all_angles[self.downsample_indices], dtype=np.float32
        )
        self.sensor_model_table = self._precompute_sensor_model()

    def _load_map(self, map_path: str) -> None:
        path = Path(map_path)
        if path.suffix in {".yaml", ".yml"}:
            yaml_path = path
            map_stem = path.with_suffix("")
        else:
            yaml_path = Path(f"{map_path}.yaml")
            map_stem = path

        with yaml_path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)

        image_name = metadata.get("image")
        image_path = yaml_path.parent / image_name if image_name else Path(f"{map_stem}.png")
        image = np.asarray(Image.open(image_path).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        if image.ndim == 3:
            image = image[..., 0]
        free_space = image > 128

        self.resolution = float(metadata["resolution"])
        self.origin_x = float(metadata["origin"][0])
        self.origin_y = float(metadata["origin"][1])
        origin_theta = float(metadata["origin"][2])
        self.origin_cos = float(np.cos(origin_theta))
        self.origin_sin = float(np.sin(origin_theta))
        self.distance_transform = np.ascontiguousarray(
            self.resolution * distance_transform_edt(free_space), dtype=np.float64
        )
        self.max_range_pixels = int(self.config.max_range / self.resolution)

    def _precompute_sensor_model(self) -> np.ndarray:
        width = self.max_range_pixels + 1
        table = np.zeros((width, width), dtype=np.float64)
        pixel_values = np.arange(width, dtype=np.float64)

        for expected in range(width):
            difference = pixel_values - expected
            probability = (
                self.config.z_hit
                * np.exp(-(difference * difference) / (2.0 * self.config.sigma_hit**2))
                / (self.config.sigma_hit * np.sqrt(2.0 * np.pi))
            )
            if expected > 0:
                short_mask = pixel_values < expected
                probability[short_mask] += (
                    2.0
                    * self.config.z_short
                    * (expected - pixel_values[short_mask])
                    / expected
                )
            probability[-1] += self.config.z_max
            probability[:-1] += self.config.z_rand / self.max_range_pixels
            table[:, expected] = probability / probability.sum()

        return table

    @staticmethod
    def odometry_delta(previous_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        """Convert two map-frame odometry poses into a vehicle-local increment."""
        world_delta = current_pose[:2] - previous_pose[:2]
        cosine = np.cos(previous_pose[2])
        sine = np.sin(previous_pose[2])
        return np.array(
            [
                cosine * world_delta[0] + sine * world_delta[1],
                -sine * world_delta[0] + cosine * world_delta[1],
                _wrap_angle(current_pose[2] - previous_pose[2]),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def integrate_odometry(pose: np.ndarray, local_delta: np.ndarray) -> np.ndarray:
        """Integrate a vehicle-local odometry increment into a map-frame pose."""
        cosine = np.cos(pose[2])
        sine = np.sin(pose[2])
        return np.array(
            [
                pose[0] + cosine * local_delta[0] - sine * local_delta[1],
                pose[1] + sine * local_delta[0] + cosine * local_delta[1],
                _wrap_angle(pose[2] + local_delta[2]),
            ],
            dtype=np.float64,
        )

    def initialize_pose(self, pose: np.ndarray) -> None:
        """Initialize particles around a supplied initial pose, as the ROS node does."""
        pose = np.asarray(pose, dtype=np.float64)
        self.particles[:, 0] = pose[0] + self.rng.normal(
            0.0, self.config.init_position_std, self.config.max_particles
        )
        self.particles[:, 1] = pose[1] + self.rng.normal(
            0.0, self.config.init_position_std, self.config.max_particles
        )
        self.particles[:, 2] = pose[2] + self.rng.normal(
            0.0, self.config.init_theta_std, self.config.max_particles
        )
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
        self.weights.fill(1.0 / self.config.max_particles)
        self.last_odometry_pose = pose.copy()
        self.inferred_pose = pose.copy()

    def _motion_model(self, proposal: np.ndarray, action: np.ndarray) -> None:
        cosine = np.cos(proposal[:, 2])
        sine = np.sin(proposal[:, 2])
        proposal[:, 0] += cosine * action[0] - sine * action[1]
        proposal[:, 1] += sine * action[0] + cosine * action[1]
        proposal[:, 2] += action[2]
        proposal[:, 0] += self.rng.normal(
            0.0, self.config.motion_dispersion_x, self.config.max_particles
        )
        proposal[:, 1] += self.rng.normal(
            0.0, self.config.motion_dispersion_y, self.config.max_particles
        )
        proposal[:, 2] += self.rng.normal(
            0.0, self.config.motion_dispersion_theta, self.config.max_particles
        )
        proposal[:, 2] = (proposal[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

    def _sensor_weights(self, proposal: np.ndarray, scan: np.ndarray) -> np.ndarray:
        predicted = _ray_march_many(
            np.ascontiguousarray(proposal, dtype=np.float64),
            self.downsampled_angles,
            self.origin_x,
            self.origin_y,
            self.origin_cos,
            self.origin_sin,
            self.resolution,
            self.distance_transform,
            self.config.max_range,
            self.config.ray_epsilon,
        )
        observed = np.nan_to_num(
            scan[self.downsample_indices],
            nan=self.config.max_range,
            posinf=self.config.max_range,
            neginf=0.0,
        )
        observed = np.clip(observed, 0.0, self.config.max_range)
        predicted = np.clip(predicted, 0.0, self.config.max_range)
        observed_pixels = np.rint(observed / self.resolution).astype(np.int64)
        predicted_pixels = np.rint(predicted / self.resolution).astype(np.int64)
        observed_pixels = np.clip(observed_pixels, 0, self.max_range_pixels)
        predicted_pixels = np.clip(predicted_pixels, 0, self.max_range_pixels)

        probabilities = self.sensor_model_table[observed_pixels[None, :], predicted_pixels]
        log_weights = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)).sum(axis=1)
        log_weights /= self.config.squash_factor
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            weights.fill(1.0 / self.config.max_particles)
        else:
            weights /= total
        return weights

    def _expected_pose(self) -> np.ndarray:
        sine_mean = np.dot(np.sin(self.particles[:, 2]), self.weights)
        cosine_mean = np.dot(np.cos(self.particles[:, 2]), self.weights)
        return np.array(
            [
                np.dot(self.particles[:, 0], self.weights),
                np.dot(self.particles[:, 1], self.weights),
                np.arctan2(sine_mean, cosine_mean),
            ],
            dtype=np.float64,
        )

    def update_from_odometry(self, odometry_pose: np.ndarray, scan: np.ndarray) -> np.ndarray:
        """Apply one MCL update from an odometry pose and matching LiDAR scan."""
        if self.last_odometry_pose is None:
            raise RuntimeError("initialize_pose must be called before the first update")
        odometry_pose = np.asarray(odometry_pose, dtype=np.float64)
        scan = np.asarray(scan, dtype=np.float64)
        if scan.shape != (self.scan_beams,):
            raise ValueError(f"Expected {self.scan_beams} LiDAR beams, got {scan.shape}")

        action = self.odometry_delta(self.last_odometry_pose, odometry_pose)
        self.last_odometry_pose = odometry_pose.copy()
        proposal_indices = self.rng.choice(
            self.particle_indices,
            size=self.config.max_particles,
            replace=True,
            p=self.weights,
        )
        proposal = self.particles[proposal_indices].copy()
        self._motion_model(proposal, action)
        self.weights = self._sensor_weights(proposal, scan)
        self.particles = proposal
        self.inferred_pose = self._expected_pose()
        return self.inferred_pose.copy()

    def predicted_scan(self, poses: np.ndarray) -> np.ndarray:
        """Return the downsampled map scan expected at one or more poses."""
        poses = np.atleast_2d(np.asarray(poses, dtype=np.float64))
        return _ray_march_many(
            np.ascontiguousarray(poses),
            self.downsampled_angles,
            self.origin_x,
            self.origin_y,
            self.origin_cos,
            self.origin_sin,
            self.resolution,
            self.distance_transform,
            self.config.max_range,
            self.config.ray_epsilon,
        )
