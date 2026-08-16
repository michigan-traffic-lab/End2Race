import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ppo.env import GroupVecEnv, Scenario, collect_group
from ppo.policy import Critic, GaussianActor
from utils import (
    find_opponent_start_index,
    get_ego_idx_range,
    load_raceline,
    require_end2race_runtime,
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race with group-structured PPO")

    # Model paths
    parser.add_argument("--checkpoint_path", type=str, default="runs/b1024_lr0.0005_m1/checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="runs/ppo")

    # Environment configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--ego_raceline", type=str, default="raceline1")
    parser.add_argument("--opponent_racelines", nargs="+", default=["raceline0", "raceline1", "raceline2"])
    parser.add_argument("--opponent_speed_scales", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--num_startpoints", type=int, default=80)
    parser.add_argument("--interval_index", type=int, default=15)
    parser.add_argument("--episode_duration", type=float, default=8.0)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Training configuration
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # PPO configuration
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)

    # Exploration configuration
    parser.add_argument("--steering_std", type=float, default=0.03)
    parser.add_argument("--speed_std", type=float, default=0.15)

    return parser.parse_args()


def log_jsonl(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def build_scenario_pool(settings):
    """Build the ego-start x opponent-raceline x speed-scale matrix."""
    ego_waypoints = load_raceline(settings.map_name, f"{settings.ego_raceline}.csv")
    ego_indices = get_ego_idx_range(settings.map_name, settings.ego_raceline, settings.num_startpoints)

    scenarios = []
    for opponent_raceline in settings.opponent_racelines:
        opponent_waypoints = ego_waypoints if opponent_raceline == settings.ego_raceline else load_raceline(settings.map_name, f"{opponent_raceline}.csv")
        for opponent_speed_scale in settings.opponent_speed_scales:
            for ego_idx in ego_indices:
                opponent_idx = find_opponent_start_index(ego_waypoints, opponent_waypoints, ego_idx, settings.interval_index)
                scenario_id = f"e{ego_idx:04d}_{opponent_raceline}_s{opponent_speed_scale}"
                scenarios.append(Scenario(scenario_id, int(ego_idx), int(opponent_idx), opponent_raceline, float(opponent_speed_scale)))
    return tuple(scenarios)


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


def score_group(group, gamma, gae_lambda):
    """Attach per-step GAE advantages and returns."""
    for trajectory in group:
        generalized_advantages(trajectory, gamma, gae_lambda)
        trajectory["episode_return"] = discounted_return(trajectory["rewards"], gamma)


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


def train_networks(actor, critic, actor_optimizer, critic_optimizer, groups, args, rng, device):
    """Train once over shuffled batches of whole scenario groups."""
    statistics = {name: [] for name in ("policy_loss", "value_loss", "clip_fraction", "approx_kl", "actor_grad_norm", "critic_grad_norm")}

    order = rng.permutation(len(groups))
    for start in range(0, len(order), args.batch_size):
        selected = order[start : start + args.batch_size]
        trajectories = [trajectory for index in selected for trajectory in groups[int(index)]]
        observations, actions, old_log_probs, advantages, returns, valid = pad_batch(trajectories, device)
        mask = valid > 0

        normalized = torch.zeros_like(advantages)
        normalized[mask] = (advantages[mask] - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

        log_probs = actor.evaluate(observations, actions)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        surrogate = torch.min(ratio * normalized, torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * normalized)
        policy_loss = -surrogate[mask].mean()

        actor_optimizer.zero_grad()
        policy_loss.backward()
        statistics["actor_grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)))
        actor_optimizer.step()

        values = critic.evaluate(observations)
        value_loss = torch.nn.functional.mse_loss(values[mask], returns[mask])

        critic_optimizer.zero_grad()
        value_loss.backward()
        statistics["critic_grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)))
        critic_optimizer.step()

        with torch.no_grad():
            statistics["policy_loss"].append(float(policy_loss.item()))
            statistics["value_loss"].append(float(value_loss.item()))
            statistics["approx_kl"].append(float(((torch.exp(log_ratio) - 1) - log_ratio)[mask].mean().item()))
            statistics["clip_fraction"].append(float((torch.abs(ratio - 1) > args.clip_range)[mask].float().mean().item()))

    return statistics


def explained_variance(groups):
    """Report how much of the return variance the rollout critic already predicted."""
    values = np.concatenate([trajectory["values"] for group in groups for trajectory in group])
    returns = np.concatenate([trajectory["returns"] for group in groups for trajectory in group])
    variance = returns.var()
    if variance < 1e-6:
        return 0.0
    return float(1.0 - (returns - values).var() / variance)


def epoch_metrics(epoch, groups, statistics):
    """Summarize one epoch from its episode records and optimization statistics."""
    records = [trajectory["record"] for group in groups for trajectory in group]
    metrics = {
        "epoch": epoch,
        "trajectories": len(records),
        "transitions": int(sum(record["episode_steps"] for record in records)),
        "explained_variance": explained_variance(groups),
        "ego_collision_count": sum(record["outcome"] == "ego_collision" for record in records),
        "overtake_count": sum(record["outcome"] == "overtake" for record in records),
        "follow_count": sum(record["outcome"] == "follow" for record in records),
        "mean_episode_steps": float(np.mean([record["episode_steps"] for record in records])),
        "mean_episode_return": float(np.mean([trajectory["episode_return"] for group in groups for trajectory in group])),
    }
    for name, values in statistics.items():
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_max"] = float(np.max(values))
    return metrics


def main():
    args = parse_arguments()
    require_end2race_runtime()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    scenarios = build_scenario_pool(args)
    print(f"Scenario pool: {len(scenarios)} | scenarios per epoch: {len(scenarios)} | group size: {args.num_envs} | epochs: {args.epochs} | batch size: {args.batch_size} scenarios | device: {device}")

    actor = GaussianActor(args.checkpoint_path, args.steering_std, args.speed_std).to(device)
    critic = Critic(args.checkpoint_path).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    scenario_rng = np.random.default_rng(args.seed)
    batch_rng = np.random.default_rng(args.seed + 1)
    vector_env = GroupVecEnv(args.num_envs, args)

    try:
        for epoch in range(1, args.epochs + 1):
            groups = []
            for scenario_index in scenario_rng.permutation(len(scenarios)):
                scenario = scenarios[int(scenario_index)]
                group = collect_group(vector_env, actor, critic, scenario, device)
                score_group(group, args.gamma, args.gae_lambda)
                groups.append(group)
                for trajectory in group:
                    log_jsonl(output_dir / "episodes.jsonl", {"epoch": epoch, **trajectory["record"]})

            statistics = train_networks(actor, critic, actor_optimizer, critic_optimizer, groups, args, batch_rng, device)
            metrics = epoch_metrics(epoch, groups, statistics)
            log_jsonl(output_dir / "metrics.jsonl", metrics)
            torch.save(actor.model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch}/{args.epochs} | collisions {metrics['ego_collision_count']} | overtakes {metrics['overtake_count']} | return {metrics['mean_episode_return']:.4f} | ev {metrics['explained_variance']:.3f} | kl {metrics['approx_kl_mean']:.2e} | gnorm {metrics['actor_grad_norm_mean']:.2f}")
    finally:
        vector_env.close()

    print("Training completed successfully")


if __name__ == "__main__":
    main()
