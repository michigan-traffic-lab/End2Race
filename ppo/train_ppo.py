import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ppo.env import GroupVecEnv
from ppo.policy import Critic, GaussianActor
from ppo.rollout import collect_group, explained_variance, score_group, train_networks
from ppo.scenarios import build_scenario_pool, scenario_stream
from utils import load_racetrack_config, load_yaml_config, require_end2race_runtime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race with group-structured PPO")

    # Model paths
    parser.add_argument("--checkpoint_path", type=str, default="runs/b1024_lr0.0005_m1/checkpoint.pt")
    parser.add_argument("--output_dir", type=str, default="runs/ppo")

    # Environment configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Rollout configuration
    parser.add_argument("--groups_per_update", type=int, default=16)
    parser.add_argument("--num_updates", type=int, default=45)

    # Training configuration
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup_updates", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--groups_per_minibatch", type=int, default=4)
    parser.add_argument("--gru_learning_rate", type=float, default=3.0e-6)
    parser.add_argument("--head_learning_rate", type=float, default=3.0e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # PPO configuration
    parser.add_argument("--advantage", type=str, default="gae")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=None)

    return parser.parse_args()


def log_jsonl(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def update_metrics(update, groups, spreads, statistics, minimum_spread):
    """Summarize one update from its episode records and optimization statistics."""
    records = [trajectory["record"] for group in groups for trajectory in group]
    metrics = {
        "update": update,
        "trajectories": len(records),
        "transitions": int(sum(record["episode_steps"] for record in records)),
        "explained_variance": explained_variance(groups),
        "ego_collision_count": sum(record["outcome"] == "ego_collision" for record in records),
        "overtake_count": sum(record["outcome"] == "overtake" for record in records),
        "follow_count": sum(record["outcome"] == "follow" for record in records),
        "mean_episode_steps": float(np.mean([record["episode_steps"] for record in records])),
        "mean_episode_return": float(np.mean([trajectory["episode_return"] for group in groups for trajectory in group])),
        "mean_group_spread": float(np.mean(spreads)),
        "silent_groups": int(sum(spread < minimum_spread for spread in spreads)),
    }
    for name, values in statistics.items():
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_max"] = float(np.max(values))
    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    require_end2race_runtime()
    if args.advantage not in {"gae", "group"}:
        raise ValueError("advantage must be gae or group")
    config = load_yaml_config(Path("ppo/config.yaml"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl(output_dir / "setup.jsonl", {"args": vars(args), "config": {name: vars(section) for name, section in vars(config).items()}})

    scenarios = build_scenario_pool(config.simulation)
    stream = scenario_stream(scenarios, args.seed)
    passes = args.groups_per_update * args.num_updates / len(scenarios)
    print(f"Scenario pool: {len(scenarios)} | group size: {args.num_envs} | groups per update: {args.groups_per_update} | advantage: {args.advantage} | pool passes: {passes:.2f} | device: {device}")

    actor = GaussianActor(args.checkpoint_path, config.exploration.steering_std, config.exploration.speed_std).to(device)
    critic = Critic(load_racetrack_config().vehicle.maximum_speed).to(device)
    # The BC-pretrained GRU moves an order of magnitude slower than the freshly adapted action head
    gru_parameters = set(actor.model.gru.parameters())
    actor_optimizer = torch.optim.Adam([
        {"params": list(gru_parameters), "lr": args.gru_learning_rate},
        {"params": [parameter for parameter in actor.parameters() if parameter not in gru_parameters], "lr": args.head_learning_rate},
    ])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    minibatch_rng = np.random.default_rng(args.seed + 1)
    vector_env = GroupVecEnv(args.num_envs, args.map_name, config)

    try:
        for update in range(1, args.num_updates + 1):
            groups = []
            spreads = []
            for _ in range(args.groups_per_update):
                scenario = next(stream)
                group = collect_group(vector_env, actor, critic, scenario, device)
                spreads.append(score_group(group, args.gamma, args.gae_lambda, args.advantage, config.advantage.minimum_group_spread))
                groups.append(group)
                for trajectory in group:
                    log_jsonl(output_dir / "episodes.jsonl", {"update": update, **trajectory["record"]})

            # The critic starts from scratch, so hold the BC actor still until its baseline is fitted
            warmup = update <= args.warmup_updates
            statistics = train_networks(actor, critic, actor_optimizer, critic_optimizer, groups, args, minibatch_rng, device, args.warmup_epochs if warmup else args.epochs, not warmup)
            metrics = update_metrics(update, groups, spreads, statistics, config.advantage.minimum_group_spread)
            metrics["warmup"] = bool(warmup)
            log_jsonl(output_dir / "metrics.jsonl", metrics)
            torch.save(actor.model.state_dict(), output_dir / f"update{update}.pt")
            torch.save(critic.state_dict(), output_dir / f"critic{update}.pt")
            print(f"Update {update}/{args.num_updates}{' (critic warmup)' if warmup else ''} | collisions {metrics['ego_collision_count']} | overtakes {metrics['overtake_count']} | return {metrics['mean_episode_return']:.4f} | ev {metrics['explained_variance']:.3f} | kl {metrics['approx_kl_mean']:.2e} | gnorm {metrics['actor_grad_norm_mean']:.2f}")
    finally:
        vector_env.close()

    print("Training completed successfully")
