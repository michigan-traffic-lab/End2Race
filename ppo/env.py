import multiprocessing as mp
import os
import traceback
import f110_gym  # Registers the F1TENTH Gym environment.
import gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from latticeplanner.lattice_planner import create_opponent
from model import End2Race
from ppo.reward import transition_reward, wrapped_progress_delta
from utils import *

OBSERVATION_SIZE = End2Race.NUM_LIDAR_FEATURES + 1


class RaceEnv:
    """Two-agent racing episode stepped at 40 Hz over 120 Hz physics."""

    def __init__(self, map_name, config):
        self.map_name = map_name
        self.config = config
        self.vehicle = load_racetrack_config().vehicle
        self.control_steps = int(round(config.simulation.episode_duration * CONTROL_FREQUENCY_HZ))

        self.env = gym.make("f110-v0", map=f"f1tenth_racetracks/{map_name}/{map_name}_map", map_ext=".png", num_agents=2, timestep=SIMULATION_TIMESTEP, integrator=Integrator.RK4)
        self.ego_waypoints = load_raceline(map_name, f"{config.simulation.ego_raceline}.csv")
        self.opponent_waypoints = {}
        self.opponents = {}

        # Progress is measured against the ego raceline for both vehicles
        self.centerline = self.ego_waypoints[:, :2]
        self.track_length = float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())

    def _opponent(self, raceline):
        if raceline not in self.opponents:
            self.opponents[raceline] = create_opponent(self.map_name, raceline)[0]
            self.opponent_waypoints[raceline] = load_raceline(self.map_name, f"{raceline}.csv")
        return self.opponents[raceline]

    def _progress(self, observation, agent):
        position = np.array([observation["poses_x"][agent], observation["poses_y"][agent]])
        return project_point_to_centerline(position, self.centerline)[0]

    def _observation(self, observation):
        lidar = downsample_lidar(observation["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES)
        return np.concatenate((lidar, [self.previous_speed])).astype(np.float32)

    def reset(self, scenario):
        """Place both vehicles for one scenario and return the first observation."""
        self._opponent(scenario.opponent_raceline)
        opponent_waypoints = self.opponent_waypoints[scenario.opponent_raceline]
        positions = np.array([
            self.ego_waypoints[scenario.ego_idx % len(self.ego_waypoints), :3],
            opponent_waypoints[scenario.opponent_idx % len(opponent_waypoints), :3],
        ])
        initial_speed = EGO_INITIAL_SPEED_FRACTION * self.vehicle.maximum_speed
        opponent_speed = opponent_waypoints[scenario.opponent_idx % len(opponent_waypoints), 3] * scenario.opponent_speed_scale
        velocities = np.array([initial_speed, opponent_speed])

        self.scenario = scenario
        self.raw_observation = self.env.reset(poses=positions, velocities=velocities)[0]
        self.previous_speed = initial_speed
        self.elapsed_time = 0.0
        self.episode_steps = 0
        self.episode_return = 0.0
        self.tracker_count = 0
        self.opponent_trajectory = None
        self.ego_progress = self._progress(self.raw_observation, 0)
        self.opponent_progress = self._progress(self.raw_observation, 1)
        self.relative_position = wrapped_progress_delta(self.ego_progress, self.opponent_progress, self.track_length)
        return self._observation(self.raw_observation)

    def _opponent_action(self):
        opponent = self.opponents[self.scenario.opponent_raceline]
        if self.tracker_count == 0:
            self.opponent_trajectory = opponent.plan(
                self.raw_observation["poses_x"][1],
                self.raw_observation["poses_y"][1],
                self.raw_observation["poses_theta"][1],
                self.raw_observation["scans"][1],
                self.raw_observation["linear_vels_x"][1],
            )
        steering, speed = opponent.tracker.plan(
            self.raw_observation["poses_x"][1],
            self.raw_observation["poses_y"][1],
            self.raw_observation["poses_theta"][1],
            self.raw_observation["linear_vels_x"][1],
            self.opponent_trajectory,
        )
        steering = np.clip(steering, -self.vehicle.steering_limit, self.vehicle.steering_limit)
        return steering, speed * self.scenario.opponent_speed_scale

    def step(self, action):
        """Hold one 40 Hz action across three physics steps and score the interval."""
        ego_steering = float(np.clip(action[0], -self.vehicle.steering_limit, self.vehicle.steering_limit))
        ego_speed = float(np.clip(action[1], 0.0, self.vehicle.maximum_speed))
        self.previous_speed = float(self.raw_observation["linear_vels_x"][0])

        ego_collision = False
        base_done = False
        for _ in range(SIMULATION_STEPS_PER_CONTROL):
            opponent_steering, opponent_speed = self._opponent_action()
            joint_action = np.array([[ego_steering, ego_speed], [opponent_steering, opponent_speed]])
            self.raw_observation, timestep, base_done, _ = self.env.step(joint_action)
            self.elapsed_time += timestep
            self.tracker_count = (self.tracker_count + 1) % self.opponents[self.scenario.opponent_raceline].conf.tracker_steps
            if self.raw_observation["collisions"][0]:
                ego_collision = True
            if ego_collision or base_done:
                break

        # Score progress across the whole control interval
        ego_progress = self._progress(self.raw_observation, 0)
        opponent_progress = self._progress(self.raw_observation, 1)
        ego_delta = wrapped_progress_delta(ego_progress, self.ego_progress, self.track_length)
        opponent_delta = wrapped_progress_delta(opponent_progress, self.opponent_progress, self.track_length)
        self.ego_progress = ego_progress
        self.opponent_progress = opponent_progress
        self.relative_position += ego_delta - opponent_delta

        reward = transition_reward(ego_delta, opponent_delta, ego_collision, self.config.reward)
        self.episode_return += reward
        self.episode_steps += 1

        timeout = self.episode_steps >= self.control_steps
        done = ego_collision or base_done or timeout
        info = {}
        if done:
            if ego_collision:
                outcome = "ego_collision"
            elif self.relative_position > 0.0:
                outcome = "overtake"
            else:
                outcome = "follow"
            info = {
                "scenario_id": self.scenario.scenario_id,
                "outcome": outcome,
                "episode_return": float(self.episode_return),
                "episode_steps": int(self.episode_steps),
                "elapsed_time": float(self.elapsed_time),
                "relative_position": float(self.relative_position),
                "ego_collision": bool(ego_collision),
                "timeout": bool(timeout),
            }
        return self._observation(self.raw_observation), reward, done, info

    def close(self):
        self.env.close()


def _worker(remote, parent_remote, map_name, config):
    parent_remote.close()
    env = None
    try:
        env = RaceEnv(map_name, config)
        while True:
            command, data = remote.recv()
            if command == "reset":
                remote.send(("ok", env.reset(data)))
            elif command == "step":
                remote.send(("ok", env.step(data)))
            elif command == "close":
                break
            else:
                raise RuntimeError(f"Unknown command {command}")
    except (KeyboardInterrupt, EOFError):
        pass
    except BaseException:
        remote.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        remote.close()


class GroupVecEnv:
    """Run one scenario across every worker so trajectories differ only by action noise."""

    def __init__(self, num_envs, map_name, config, start_method="forkserver"):
        self.num_envs = num_envs
        self.closed = False
        context = mp.get_context(start_method)
        self.remotes, work_remotes = zip(*[context.Pipe() for _ in range(num_envs)])
        self.processes = []

        previous_threads = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        for name in previous_threads:
            os.environ[name] = "1"
        try:
            for work_remote, remote in zip(work_remotes, self.remotes):
                process = context.Process(target=_worker, args=(work_remote, remote, map_name, config), daemon=True)
                process.start()
                self.processes.append(process)
                work_remote.close()
        except BaseException:
            self.close()
            raise
        finally:
            for name, value in previous_threads.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _receive(self, rank):
        status, payload = self.remotes[rank].recv()
        if status != "ok":
            self.close()
            raise RuntimeError(f"environment worker {rank} failed:\n{payload}")
        return payload

    def reset_group(self, scenario):
        """Broadcast one scenario to every worker and return stacked observations."""
        for remote in self.remotes:
            remote.send(("reset", scenario))
        return np.stack([self._receive(rank) for rank in range(self.num_envs)])

    def step(self, actions, active):
        """Step only the workers still running and return per-rank results."""
        ranks = [rank for rank in range(self.num_envs) if active[rank]]
        for rank in ranks:
            self.remotes[rank].send(("step", actions[rank]))
        results = [None] * self.num_envs
        for rank in ranks:
            results[rank] = self._receive(rank)
        return results

    def close(self):
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
