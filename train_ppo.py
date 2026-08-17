import time

import numpy as np
import torch

from ppo.env import collect_batch

INITIAL_STEERING_STD = 0.05
INITIAL_SPEED_STD = 0.50
UPDATE_EPOCHS = 4
VALUE_LOSS_WEIGHT = 0.5


def discounted_return(rewards, gamma):
    if gamma == 1.0:
        return float(rewards.sum())
    return float((gamma ** np.arange(len(rewards), dtype=np.float64) * rewards).sum())


def generalized_advantages(trajectory, gamma, gae_lambda):
    """Accumulate the GAE recursion backwards over one trajectory."""
    rewards = trajectory["rewards"].astype(np.float64)
    values = trajectory["values"].astype(np.float64)
    advantages = np.zeros_like(rewards)
    running = 0.0
    next_value = trajectory["bootstrap"]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        running = delta + gamma * gae_lambda * running
        advantages[step] = running
        next_value = values[step]
    trajectory["advantages"] = advantages.astype(np.float32)
    trajectory["returns"] = (advantages + values).astype(np.float32)


def score_batch(batch, gamma, gae_lambda):
    """Attach per-step GAE advantages and returns."""
    for trajectory in batch:
        generalized_advantages(trajectory, gamma, gae_lambda)
        trajectory["episode_return"] = discounted_return(trajectory["rewards"], gamma)


def rollout_diagnostics(batch):
    records = [trajectory["record"] for trajectory in batch]
    policy_means = np.concatenate([trajectory["policy_means"] for trajectory in batch])
    values = np.concatenate([trajectory["values"] for trajectory in batch])
    collisions = sum(record["ego_collision"] for record in records)
    return {
        "collisions": collisions,
        "follows": sum(record["outcome"] == "follow" for record in records),
        "overtakes": sum(record["outcome"] == "overtake" for record in records),
        "steering_mean": float(policy_means[:, 0].mean()),
        "steering_std": float(policy_means[:, 0].std()),
        "speed_mean": float(policy_means[:, 1].mean()),
        "speed_std": float(policy_means[:, 1].std()),
        "value_mean": float(values.mean()),
        "episode_return_mean": float(np.mean([trajectory["episode_return"] for trajectory in batch])),
    }


def pad_batch(trajectories, device):
    """Pad a set of variable-length trajectories into one padded tensor batch."""
    lengths = [len(trajectory["rewards"]) for trajectory in trajectories]
    max_length = max(lengths)
    count = len(trajectories)
    observation_size = trajectories[0]["observations"].shape[1]

    observations = np.zeros((count, max_length, observation_size), dtype=np.float32)
    actions = np.zeros((count, max_length, 2), dtype=np.float32)
    log_probs = np.zeros((count, max_length), dtype=np.float32)
    advantages = np.zeros((count, max_length), dtype=np.float32)
    returns = np.zeros((count, max_length), dtype=np.float32)
    valid = np.zeros((count, max_length), dtype=np.float32)
    for slot, trajectory in enumerate(trajectories):
        length = lengths[slot]
        observations[slot, :length] = trajectory["observations"]
        actions[slot, :length] = trajectory["actions"]
        log_probs[slot, :length] = trajectory["log_probs"]
        advantages[slot, :length] = trajectory["advantages"]
        returns[slot, :length] = trajectory["returns"]
        valid[slot, :length] = 1.0

    return tuple(torch.as_tensor(array, device=device) for array in (observations, actions, log_probs, advantages, returns, valid))


def train_batch(model, optimizer, trajectories, args, device):
    """Update from one shuffled batch containing one trajectory per scenario."""
    observations, actions, old_log_probs, advantages, returns, valid = pad_batch(trajectories, device)
    mask = valid > 0

    normalized = torch.zeros_like(advantages)
    normalized[mask] = (advantages[mask] - advantages[mask].mean()) / (advantages[mask].std(unbiased=False) + 1e-8)

    log_probs, values, _ = model.evaluate(observations, actions)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    surrogate = torch.min(
        ratio * normalized,
        torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
        * normalized,
    )
    policy_loss = -surrogate[mask].mean()
    value_loss = torch.nn.functional.mse_loss(values[mask], returns[mask])
    loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
    optimizer.step()

    with torch.no_grad():
        new_log_probs, _, _ = model.evaluate(observations, actions)
        new_log_ratio = new_log_probs - old_log_probs
        new_ratio = torch.exp(new_log_ratio)
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "approx_kl": float(((new_ratio - 1) - new_log_ratio)[mask].mean().item()),
            "clip_fraction": float((torch.abs(new_ratio - 1) > args.clip_range)[mask].float().mean().item()),
            "grad_norm": grad_norm,
        }


def explained_variance(batches):
    """Report how much of the return variance the rollout value function predicted."""
    values = np.concatenate([trajectory["values"] for batch in batches for trajectory in batch])
    returns = np.concatenate([trajectory["returns"] for batch in batches for trajectory in batch])
    variance = returns.var()
    if variance < 1e-6:
        return 0.0
    return float(1.0 - (returns - values).var() / variance)


def train_epoch(model, optimizer, envs, scenarios, rng, args, device, epoch):
    batches = []
    statistics = {name: [] for name in ("policy_loss", "value_loss", "clip_fraction", "approx_kl", "grad_norm")}
    if not scenarios:
        print(f"Epoch {epoch} rollouts: no scenarios", flush=True)
        return [], batches, statistics

    order = rng.permutation(len(scenarios))
    ordered = [scenarios[int(index)] for index in order]
    started_at = time.monotonic()
    print(f"Epoch {epoch} rollouts: 0/{len(ordered)} scenarios", flush=True)
    for start in range(0, len(ordered), envs.num_envs):
        selected = ordered[start : start + envs.num_envs]
        batch = collect_batch(envs, model, selected, device)
        score_batch(batch, args.gamma, args.gae_lambda)
        batches.append(batch)
        diagnostics = rollout_diagnostics(batch)
        for _ in range(UPDATE_EPOCHS):
            batch_statistics = train_batch(model, optimizer, batch, args, device)
            for name, value in batch_statistics.items():
                statistics[name].append(value)

        elapsed = time.monotonic() - started_at
        completed = start + len(selected)
        steering_std, speed_std = model.action_std.detach().cpu().tolist()
        print(
            f"Epoch {epoch} rollouts: {completed}/{len(ordered)} "
            f"| collision {diagnostics['collisions']} follow {diagnostics['follows']} "
            f"overtake {diagnostics['overtakes']} | "
            f"policy steer {diagnostics['steering_mean']:.3f}±{diagnostics['steering_std']:.3f} "
            f"speed {diagnostics['speed_mean']:.3f}±{diagnostics['speed_std']:.3f} | "
            f"value {diagnostics['value_mean']:.3f} return {diagnostics['episode_return_mean']:.3f} | "
            f"std ({steering_std:.4f}, {speed_std:.4f}) | elapsed {elapsed:.0f}s",
            flush=True,
        )
    return ordered, batches, statistics
