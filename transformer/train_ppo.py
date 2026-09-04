import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from reinforcement.env import RaceEnv, VectorEnv, collect_batch, shard_scenarios
from reinforcement.eval_ppo import evaluate_scenarios
from reinforcement.run_ppo import (
    LEARNING_RATE,
    MINIMUM_OVERTAKE_RATE,
    MINIMUM_SAFETY_RATE,
    _append_record,
    _build_scenarios,
    _initialize_process_group,
    _resolved_config,
    _summarize,
    _synchronize_model,
)
from reinforcement.train_ppo import (
    ACTION_STD_DECAY,
    INITIAL_SPEED_STD,
    INITIAL_STEERING_STD,
    _gather_rollouts,
    _score_batch,
    _update_policy,
)
from transformer.policy import TransformerActorCritic

ARTIFACT_DIR = Path("checkpoint/transformer_ppo")
EPISODES_PER_UPDATE = 24


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run End2Race Transformer PPO training")

    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoint/transformer.pt"))
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--ego_raceline", type=str, default="raceline1")
    parser.add_argument("--opponent_racelines", nargs="+", default=["raceline0", "raceline1", "raceline2"])
    parser.add_argument("--opponent_speed_scales", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--num_startpoints", type=int, default=80)
    parser.add_argument("--interval_index", type=int, default=15)
    parser.add_argument("--episode_duration", type=float, default=8.0)
    parser.add_argument("--num_envs", type=int, default=8)

    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.2)

    args = parser.parse_args()
    positive_values = {
        "--num_startpoints": args.num_startpoints,
        "--episode_duration": args.episode_duration,
        "--num_envs": args.num_envs,
        "--max_grad_norm": args.max_grad_norm,
        "--clip_range": args.clip_range,
    }
    invalid_values = [name for name, value in positive_values.items() if value <= 0]
    if invalid_values:
        parser.error(f"{', '.join(invalid_values)} must be positive")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("--gamma must be in (0, 1] and --gae_lambda must be in [0, 1]")
    if args.interval_index < 0 or any(scale <= 0 for scale in args.opponent_speed_scales):
        parser.error("--interval_index must be nonnegative and opponent speed scales must be positive")
    if (
        len(set(args.opponent_racelines)) != len(args.opponent_racelines)
        or len(set(args.opponent_speed_scales)) != len(args.opponent_speed_scales)
    ):
        parser.error("opponent racelines and speed scales must not contain duplicates")
    if not args.checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint_path}")
    return args


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
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if rank == 0:
        order = torch.as_tensor(rng.permutation(len(scenarios)), device=device)
    else:
        order = torch.empty(len(scenarios), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.broadcast(order, src=0)
    ordered = [scenarios[int(index)] for index in order.cpu().numpy()]
    scenario_batches = [
        ordered[start : start + EPISODES_PER_UPDATE]
        for start in range(0, len(ordered), EPISODES_PER_UPDATE)
    ]

    if rank == 0:
        print(f"Epoch {epoch} rollouts: 0/{len(ordered)} scenarios", flush=True)
    batches = []
    for update, scenario_batch in enumerate(scenario_batches, start=1):
        local_scenarios = shard_scenarios(scenario_batch, world_size)[rank]
        update_batches = []
        for start in range(0, len(local_scenarios), envs.num_envs):
            selected = local_scenarios[start : start + envs.num_envs]
            batch = collect_batch(envs, model, selected, device)
            _score_batch(batch, args.gamma, args.gae_lambda)
            update_batches.extend([trajectory] for trajectory in batch)
            batches.append(batch)

        completed = min(update * EPISODES_PER_UPDATE, len(ordered))
        if rank == 0:
            print(
                f"Epoch {epoch} scenarios {completed}/{len(ordered)} collected",
                flush=True,
            )
        update_statistics = _update_policy(
            model,
            optimizer,
            update_batches,
            args,
            device,
            epoch,
            update,
            len(scenario_batches),
        )
        for name, value in update_statistics.items():
            statistics[name].append(value)

    if rank == 0:
        print(
            f"Epoch {epoch} rollouts complete: {len(ordered)}/{len(ordered)}",
            flush=True,
        )
    return ordered, _gather_rollouts(batches), statistics


def main():
    args = parse_arguments()
    rank, world_size, device = _initialize_process_group()
    envs = None
    try:
        config_path = ARTIFACT_DIR / "config.json"
        episodes_path = ARTIFACT_DIR / "episodes.jsonl"
        metrics_path = ARTIFACT_DIR / "metrics.jsonl"
        artifact_error = None
        if rank == 0:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            existing = sorted(path.name for path in ARTIFACT_DIR.iterdir())
            if existing:
                artifact_error = (
                    f"Cannot start PPO in {ARTIFACT_DIR}: existing "
                    f"{', '.join(existing)}"
                )
        if dist.is_initialized():
            errors = [artifact_error]
            dist.broadcast_object_list(errors, src=0)
            artifact_error = errors[0]
        if artifact_error:
            raise FileExistsError(artifact_error)

        scenarios = _build_scenarios(args)
        if world_size > len(scenarios):
            raise ValueError(
                f"Cannot distribute {len(scenarios)} scenarios across "
                f"{world_size} GPU processes"
            )
        model = TransformerActorCritic(
            args.checkpoint_path,
            INITIAL_STEERING_STD,
            INITIAL_SPEED_STD,
        ).to(device)
        _synchronize_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if rank == 0:
            config = _resolved_config(args, len(scenarios), world_size)
            config["model"] = "transformer"
            config["context_length"] = model.model.CONTEXT_LENGTH
            config["batch_size"] = EPISODES_PER_UPDATE
            config["updates_per_epoch"] = len(scenarios) // EPISODES_PER_UPDATE
            config["update_epochs"] = 1
            config_path.write_text(
                json.dumps(config, indent=2) + "\n",
                encoding="utf-8",
            )
            episodes_path.touch(exist_ok=True)
            metrics_path.touch(exist_ok=True)
            print(
                f"Scenario pool: {len(scenarios)} | GPU processes: {world_size} | "
                f"workers/GPU: {args.num_envs} | "
                f"total workers: {args.num_envs * world_size} | "
                f"lr: {LEARNING_RATE:.1e}"
            )

        envs = VectorEnv(args.num_envs, args)
        rng = np.random.default_rng()
        saved_model_count = 0
        for epoch in itertools.count(1):
            epoch_action_std = model.action_std.detach().cpu().tolist()
            ordered, rollout_summary, statistics = train_epoch(
                model,
                optimizer,
                envs,
                scenarios,
                rng,
                args,
                device,
                epoch,
            )
            if rank == 0:
                for trajectory in rollout_summary["trajectories"]:
                    _append_record(
                        episodes_path,
                        {"epoch": epoch, "phase": "training", **trajectory["record"]},
                    )

            label = f"Epoch {epoch}"
            eval_records = evaluate_scenarios(envs, model, scenarios, device, label)
            model.action_std.mul_(ACTION_STD_DECAY)
            next_action_std = model.action_std.detach().cpu().tolist()
            if rank == 0:
                for record in eval_records:
                    _append_record(
                        episodes_path,
                        {"epoch": epoch, "phase": "screening", **record},
                    )

                metrics = _summarize(
                    epoch,
                    ordered,
                    rollout_summary,
                    statistics,
                    eval_records,
                )
                metrics["exploration_steering_std"] = epoch_action_std[0]
                metrics["exploration_speed_std"] = epoch_action_std[1]
                metrics["next_exploration_steering_std"] = next_action_std[0]
                metrics["next_exploration_speed_std"] = next_action_std[1]
                model_saved = (
                    metrics["screening_safety_rate"] > MINIMUM_SAFETY_RATE
                    and metrics["screening_overtake_rate"] > MINIMUM_OVERTAKE_RATE
                )
                metrics["model_saved"] = model_saved
                if model_saved:
                    saved_model_count += 1
                    model_path = ARTIFACT_DIR / f"ppo_{saved_model_count:03d}.pt"
                    with model_path.open("xb") as stream:
                        torch.save(model.model.state_dict(), stream)
                    print(
                        f"{label} saved {model_path.name}: "
                        f"safety {metrics['screening_safety_rate']:.2%} | "
                        f"overtake {metrics['screening_overtake_rate']:.2%}",
                        flush=True,
                    )
                _append_record(metrics_path, metrics)
                print(
                    f"{label} | safety {metrics['screening_safe_count']}/{metrics['screening_count']} "
                    f"({metrics['screening_safety_rate']:.2%}) | "
                    f"collision {metrics['screening_collision_count']}/{metrics['screening_count']} "
                    f"({metrics['screening_collision_rate']:.2%}) | "
                    f"overtake {metrics['screening_overtake_count']}/{metrics['screening_count']} "
                    f"({metrics['screening_overtake_rate']:.2%}) | model saved {model_saved} | "
                    f"training scenarios {metrics['training_scenario_count']} | "
                    f"collisions {metrics['training_collision_count']} | "
                    f"follow {metrics['training_follow_count']} | "
                    f"overtakes {metrics['training_overtake_count']} | "
                    f"return {metrics['training_mean_return']:.4f} | "
                    f"exploration std steer {epoch_action_std[0]:.6f}->{next_action_std[0]:.6f} "
                    f"speed {epoch_action_std[1]:.6f}->{next_action_std[1]:.6f}"
                )
            if dist.is_initialized():
                dist.barrier()
    finally:
        if envs is not None:
            envs.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
