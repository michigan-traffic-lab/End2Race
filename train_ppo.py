import time

import numpy as np
import torch
import torch.distributed as dist

from ppo.env import collect_batch, shard_scenarios

FIXED_STEERING_STD = 0.05
FIXED_SPEED_STD = 0.50
UPDATE_EPOCHS = 1
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

    return tuple(
        torch.as_tensor(array, device=device)
        for array in (
            observations,
            actions,
            log_probs,
            advantages,
            returns,
            valid,
        )
    )


def accumulate_batch_gradients(
    model,
    trajectories,
    args,
    device,
    advantage_mean,
    advantage_std,
    total_transitions,
):
    """Accumulate one rollout chunk's contribution to a full-pool update."""
    observations, actions, old_log_probs, advantages, returns, valid = pad_batch(trajectories, device)
    mask = valid > 0

    normalized = torch.zeros_like(advantages)
    normalized[mask] = (advantages[mask] - advantage_mean) / (advantage_std + 1e-8)

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
    weight = int(mask.sum().item()) / total_transitions
    loss = weight * (policy_loss + VALUE_LOSS_WEIGHT * value_loss)
    loss.backward()
    return float(policy_loss.item()) * weight, float(value_loss.item()) * weight


def _sum_across_ranks(values, device):
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    return totals.cpu().tolist()


def _synchronize_gradients(model):
    if not dist.is_initialized():
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)


def update_full_pool(model, optimizer, batches, args, device, epoch, update):
    """Take one optimizer step from every trajectory in the scenario pool."""
    advantage_values = np.concatenate([
        trajectory["advantages"]
        for batch in batches
        for trajectory in batch
    ])
    total_transitions, advantage_sum, advantage_square_sum = _sum_across_ranks(
        [
            len(advantage_values),
            advantage_values.sum(),
            np.square(advantage_values).sum(),
        ],
        device,
    )
    advantage_mean = advantage_sum / total_transitions
    advantage_variance = max(
        advantage_square_sum / total_transitions - advantage_mean**2,
        0.0,
    )
    advantage_std = advantage_variance**0.5
    total_transitions = int(total_transitions)

    optimizer.zero_grad()
    policy_loss = 0.0
    value_loss = 0.0
    for batch in batches:
        batch_policy_loss, batch_value_loss = accumulate_batch_gradients(
            model,
            batch,
            args,
            device,
            advantage_mean,
            advantage_std,
            total_transitions,
        )
        policy_loss += batch_policy_loss
        value_loss += batch_value_loss

    _synchronize_gradients(model)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
    applied_grad_norm = min(grad_norm, args.max_grad_norm)
    optimizer.step()

    approx_kl = 0.0
    clip_fraction = 0.0
    with torch.no_grad():
        for batch in batches:
            observations, actions, old_log_probs, _, _, valid = pad_batch(batch, device)
            mask = valid > 0
            new_log_probs, _, _ = model.evaluate(observations, actions)
            new_log_ratio = new_log_probs - old_log_probs
            new_ratio = torch.exp(new_log_ratio)
            weight = int(mask.sum().item()) / total_transitions
            approx_kl += weight * float(((new_ratio - 1) - new_log_ratio)[mask].mean().item())
            clip_fraction += weight * float(
                (torch.abs(new_ratio - 1) > args.clip_range)[mask]
                .float()
                .mean()
                .item()
            )

    policy_loss, value_loss, approx_kl, clip_fraction = _sum_across_ranks(
        [policy_loss, value_loss, approx_kl, clip_fraction],
        device,
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch} ppo update {update}/{UPDATE_EPOCHS} | "
            f"transitions {total_transitions} | lr {learning_rate:.1e} | "
            f"value loss {value_loss:.4f} | "
            f"kl {approx_kl:.6f} clip {clip_fraction:.3f} | "
            f"grad preclip {grad_norm:.3f} applied {applied_grad_norm:.3f}",
            flush=True,
        )

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "grad_norm": grad_norm,
    }


def _summarize_rollouts(batches):
    trajectories = []
    for batch in batches:
        for trajectory in batch:
            returns = trajectory["returns"].astype(np.float64)
            errors = returns - trajectory["values"].astype(np.float64)
            trajectories.append(
                {
                    "record": trajectory["record"],
                    "episode_return": float(trajectory["episode_return"]),
                    "return_sum": float(returns.sum()),
                    "return_square_sum": float(np.square(returns).sum()),
                    "error_sum": float(errors.sum()),
                    "error_square_sum": float(np.square(errors).sum()),
                }
            )
    return {"batch_count": len(batches), "trajectories": trajectories}


def _gather_rollouts(batches):
    local_summary = _summarize_rollouts(batches)
    if not dist.is_initialized():
        return local_summary
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local_summary, gathered, dst=0)
    if gathered is None:
        return None
    return {
        "batch_count": sum(summary["batch_count"] for summary in gathered),
        "trajectories": [
            trajectory
            for summary in gathered
            for trajectory in summary["trajectories"]
        ],
    }


def _collect_shard(envs, model, scenarios, device, args, epoch, started_at):
    batches = []
    batch_count = (len(scenarios) + envs.num_envs - 1) // envs.num_envs
    for start in range(0, len(scenarios), envs.num_envs):
        selected = scenarios[start : start + envs.num_envs]
        batch = collect_batch(envs, model, selected, device)
        score_batch(batch, args.gamma, args.gae_lambda)
        batches.append(batch)
        diagnostics = rollout_diagnostics(batch)
        elapsed = time.monotonic() - started_at
        completed = start + len(selected)
        print(
            f"Epoch {epoch} batch {len(batches)}/{batch_count} | "
            f"scenarios {completed}/{len(scenarios)} | "
            f"collision {diagnostics['collisions']} follow {diagnostics['follows']} "
            f"overtake {diagnostics['overtakes']} | "
            f"policy steer {diagnostics['steering_mean']:.3f}±{diagnostics['steering_std']:.3f} "
            f"speed {diagnostics['speed_mean']:.3f}±{diagnostics['speed_std']:.3f} | "
            f"value {diagnostics['value_mean']:.3f} return {diagnostics['episode_return_mean']:.3f} | "
            f"elapsed {elapsed:.0f}s",
            flush=True,
        )
    return batches


def train_epoch(model, optimizer, envs, scenarios, rng, args, device, epoch):
    statistics = {
        name: []
        for name in (
            "policy_loss",
            "value_loss",
            "clip_fraction",
            "approx_kl",
            "grad_norm",
        )
    }
    if not scenarios:
        print(f"Epoch {epoch} rollouts: no scenarios", flush=True)
        return [], None, statistics

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if rank == 0:
        order = torch.as_tensor(rng.permutation(len(scenarios)), device=device)
    else:
        order = torch.empty(len(scenarios), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.broadcast(order, src=0)
    order = order.cpu().numpy()
    ordered = [scenarios[int(index)] for index in order]
    shard = shard_scenarios(ordered, world_size)[rank]
    started_at = time.monotonic()
    if rank == 0:
        print(f"Epoch {epoch} rollouts: 0/{len(ordered)} scenarios", flush=True)
    batches = _collect_shard(
        envs,
        model,
        shard,
        device,
        args,
        epoch,
        started_at,
    )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(
            f"Epoch {epoch} rollouts complete: {len(ordered)}/{len(ordered)}",
            flush=True,
        )

    for update in range(1, UPDATE_EPOCHS + 1):
        update_statistics = update_full_pool(
            model,
            optimizer,
            batches,
            args,
            device,
            epoch,
            update,
        )
        for name, value in update_statistics.items():
            statistics[name].append(value)
    rollout_summary = _gather_rollouts(batches)
    return ordered, rollout_summary, statistics
