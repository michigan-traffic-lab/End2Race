from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
import multiprocessing as mp
import os
import traceback
import warnings

with redirect_stderr(StringIO()):
    import f110_gym  # Registers the F1TENTH Gym environment.
    import gym
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

from expert.controllers import RacelineFollower
from expert.utils import (
    downsample_lidar,
    project_point_to_centerline,
)
from f1tenth_sim.utils import (
    load_raceline,
    load_racetrack_config,
    racetrack_path,
    simulation_config,
)
from imitation.model import End2Race

warnings.filterwarnings(
    "ignore",
    message="Chosen integrator is RK4.*",
    category=UserWarning,
    module="f110_gym.envs.base_classes",
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ego_idx: int
    opponent_idx: int
    opponent_raceline: str
    opponent_speed_scale: float


def shard_scenarios(scenarios, shard_count):
    shard_size, remainder = divmod(len(scenarios), shard_count)
    shards = []
    start = 0
    for index in range(shard_count):
        end = start + shard_size + int(index < remainder)
        shards.append(tuple(scenarios[start:end]))
        start = end
    return tuple(shards)


def _wrapped_progress_delta(current_progress, previous_progress, track_length):
    """Measure signed progress across the cyclic lap boundary."""
    offset = current_progress - previous_progress + 0.5 * track_length
    return float(offset % track_length - 0.5 * track_length)


class RaceEnv:
    """Two-agent racing episode stepped at 40 Hz over 120 Hz physics."""

    PROGRESS_REWARD_WEIGHT = 0.01
    COLLISION_PENALTY = -2.5
    MAXIMUM_EGO_SPEED = 20.0

    def __init__(self, settings):
        self.map_name = settings.map_name
        self.vehicle = load_racetrack_config().vehicle
        self.simulation = simulation_config()
        self.control_steps = int(round(settings.episode_duration * self.simulation.control_frequency_hz))

        self.env = gym.make(
            "f110-v0",
            map=str(racetrack_path(self.map_name, f"{self.map_name}_map")),
            map_ext=".png",
            num_agents=2,
            timestep=self.simulation.timestep,
            integrator=Integrator.RK4,
        )
        self.ego_waypoints = load_raceline(self.map_name, f"{settings.ego_raceline}.csv")
        self.opponents = {}

        # Progress is measured against the ego raceline for both vehicles
        self.centerline = self.ego_waypoints[:, :2]
        self.track_length = float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())

    def _opponent(self, raceline):
        if raceline not in self.opponents:
            opponent = RacelineFollower(self.map_name, raceline)
            if opponent.conf.tracker_steps != self.simulation.steps_per_expert_plan:
                raise ValueError(
                    "expert.tracker_steps must match the number of simulation "
                    f"steps per expert plan ({self.simulation.steps_per_expert_plan})"
                )
            self.opponents[raceline] = opponent
        return self.opponents[raceline]

    def _progress(self, observation, agent):
        position = np.array([observation["poses_x"][agent], observation["poses_y"][agent]])
        return project_point_to_centerline(position, self.centerline)[0]

    def _observation(self, observation):
        lidar = downsample_lidar(observation["scans"][0], target_points=End2Race.NUM_LIDAR_FEATURES)
        return np.concatenate((lidar, [self.previous_speed])).astype(np.float32)

    def reset(self, scenario):
        """Place both vehicles for one scenario and return the first observation."""
        opponent = self._opponent(scenario.opponent_raceline)
        opponent_waypoints = opponent.waypoints
        opponent_idx = scenario.opponent_idx % len(opponent_waypoints)
        positions = np.array([
            self.ego_waypoints[scenario.ego_idx % len(self.ego_waypoints), :3],
            opponent_waypoints[opponent_idx, [0, 1, 3]],
        ])
        initial_speed = self.simulation.ego_initial_speed_fraction * self.vehicle.maximum_speed
        opponent_speed = opponent_waypoints[opponent_idx, 2] * scenario.opponent_speed_scale
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
        self.relative_position = _wrapped_progress_delta(
            self.ego_progress,
            self.opponent_progress,
            self.track_length,
        )
        self.overtaken = self.relative_position >= self.vehicle.length
        return self._observation(self.raw_observation)

    def _opponent_action(self):
        opponent = self.opponents[self.scenario.opponent_raceline]
        if self.tracker_count == 0:
            self.opponent_trajectory = opponent.reference_trajectory(
                self.raw_observation["poses_x"][1],
                self.raw_observation["poses_y"][1],
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
        ego_speed = float(np.clip(action[1], 0.0, self.MAXIMUM_EGO_SPEED))
        self.previous_speed = float(self.raw_observation["linear_vels_x"][0])

        ego_collision = False
        for _ in range(self.simulation.steps_per_control):
            opponent_steering, opponent_speed = self._opponent_action()
            joint_action = np.array([[ego_steering, ego_speed], [opponent_steering, opponent_speed]])
            self.raw_observation, timestep, _, _ = self.env.step(joint_action)
            self.elapsed_time += timestep
            tracker_steps = self.opponents[
                self.scenario.opponent_raceline
            ].conf.tracker_steps
            self.tracker_count = (self.tracker_count + 1) % tracker_steps
            if self.raw_observation["collisions"][0]:
                ego_collision = True
            if ego_collision:
                break

        # Score progress across the whole control interval
        ego_progress = self._progress(self.raw_observation, 0)
        opponent_progress = self._progress(self.raw_observation, 1)
        ego_delta = _wrapped_progress_delta(ego_progress, self.ego_progress, self.track_length)
        opponent_delta = _wrapped_progress_delta(
            opponent_progress,
            self.opponent_progress,
            self.track_length,
        )
        self.ego_progress = ego_progress
        self.opponent_progress = opponent_progress
        self.relative_position += ego_delta - opponent_delta

        reward = self.PROGRESS_REWARD_WEIGHT * ego_delta
        self.overtaken = self.overtaken or self.relative_position >= self.vehicle.length
        if ego_collision:
            reward += self.COLLISION_PENALTY
        self.episode_return += reward
        self.episode_steps += 1

        timeout = self.episode_steps >= self.control_steps
        done = ego_collision or timeout
        info = {}
        if done:
            if ego_collision:
                outcome = "ego_collision"
            elif self.overtaken:
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


def _worker(remote, parent_remote, settings):
    parent_remote.close()
    env = None
    try:
        env = RaceEnv(settings)
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


class VectorEnv:
    """Run independent racing scenarios across a fixed worker pool."""

    def __init__(self, num_envs, settings, start_method="forkserver"):
        self.num_envs = num_envs
        self.closed = False
        context = mp.get_context(start_method)
        self.remotes, work_remotes = zip(*[context.Pipe() for _ in range(num_envs)])
        self.processes = []

        previous_threads = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        }
        for name in previous_threads:
            os.environ[name] = "1"
        try:
            for work_remote, remote in zip(work_remotes, self.remotes):
                process = context.Process(target=_worker, args=(work_remote, remote, settings), daemon=True)
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

    def reset_scenarios(self, scenarios):
        """Assign one scenario to each participating worker."""
        if len(scenarios) > self.num_envs:
            raise ValueError(f"Cannot assign {len(scenarios)} scenarios to {self.num_envs} workers")
        for rank, scenario in enumerate(scenarios):
            self.remotes[rank].send(("reset", scenario))
        return np.stack([self._receive(rank) for rank in range(len(scenarios))])

    def step(self, actions, active):
        """Step only the workers still running and return per-rank results."""
        ranks = [rank for rank, is_active in enumerate(active) if is_active]
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


def collect_batch(envs, model, scenarios, device):
    """Roll one trajectory for each scenario under a frozen policy."""
    observations = envs.reset_scenarios(scenarios)
    state = model.initial_state(len(scenarios), device)
    active = [True] * len(scenarios)
    trajectories = [
        {
            "observations": [],
            "actions": [],
            "policy_means": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "bootstrap": 0.0,
        }
        for _ in scenarios
    ]
    records = [None] * len(scenarios)

    while any(active):
        with torch.no_grad():
            observation_batch = torch.as_tensor(observations, device=device)
            actions, policy_means, log_probs, values, next_state = model.act(observation_batch, state)
        actions = actions.cpu().numpy()
        policy_means = policy_means.cpu().numpy()
        log_probs = log_probs.cpu().numpy()
        values = values.cpu().numpy()

        bootstrap_ranks = []
        for rank, result in enumerate(envs.step(actions, active)):
            if result is None:
                continue
            next_observation, reward, done, info = result
            trajectory = trajectories[rank]
            trajectory["observations"].append(observations[rank].copy())
            trajectory["actions"].append(actions[rank])
            trajectory["policy_means"].append(policy_means[rank])
            trajectory["log_probs"].append(log_probs[rank])
            trajectory["values"].append(values[rank])
            trajectory["rewards"].append(reward)
            observations[rank] = next_observation
            if done:
                active[rank] = False
                records[rank] = info
                # Bootstrap the value at the time limit.
                if not info["ego_collision"]:
                    bootstrap_ranks.append(rank)

        if bootstrap_ranks:
            with torch.no_grad():
                tail_observations = torch.as_tensor(observations[bootstrap_ranks], device=device)
                tail_state = model.select_state(next_state, bootstrap_ranks)
                _, tail_values, _ = model.evaluate(
                    tail_observations[:, None], hidden=tail_state
                )
                tail_values = tail_values[:, 0].cpu().numpy()
            for slot, rank in enumerate(bootstrap_ranks):
                trajectories[rank]["bootstrap"] = float(tail_values[slot])

        state = next_state

    batch = []
    for trajectory, record in zip(trajectories, records):
        batch.append({
            "observations": np.asarray(trajectory["observations"], dtype=np.float32),
            "actions": np.asarray(trajectory["actions"], dtype=np.float32),
            "policy_means": np.asarray(trajectory["policy_means"], dtype=np.float32),
            "log_probs": np.asarray(trajectory["log_probs"], dtype=np.float32),
            "values": np.asarray(trajectory["values"], dtype=np.float32),
            "rewards": np.asarray(trajectory["rewards"], dtype=np.float32),
            "bootstrap": trajectory["bootstrap"],
            "record": record,
        })
    # Match one-step rollout estimates to cuDNN's full-sequence training path.
    _refresh_old_estimates(batch, model, device)
    return batch


def _refresh_old_estimates(batch, model, device):
    """Recompute log-probabilities and values on the batched training path."""
    lengths = [len(trajectory["rewards"]) for trajectory in batch]
    observations = np.zeros((len(batch), max(lengths), batch[0]["observations"].shape[1]), dtype=np.float32)
    actions = np.zeros((len(batch), max(lengths), 2), dtype=np.float32)
    for slot, trajectory in enumerate(batch):
        observations[slot, : lengths[slot]] = trajectory["observations"]
        actions[slot, : lengths[slot]] = trajectory["actions"]

    with torch.no_grad():
        observations = torch.as_tensor(observations, device=device)
        log_probs, values, _ = model.evaluate(observations, torch.as_tensor(actions, device=device))
        log_probs = log_probs.cpu().numpy()
        values = values.cpu().numpy()
    for slot, trajectory in enumerate(batch):
        trajectory["log_probs"] = log_probs[slot, : lengths[slot]].astype(np.float32)
        trajectory["values"] = values[slot, : lengths[slot]].astype(np.float32)
