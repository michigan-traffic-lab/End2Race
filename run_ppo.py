import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ppo.env import RaceEnv, Scenario, VectorEnv
from eval_ppo import evaluate_scenarios
from ppo.policy import ActorCritic
from train_ppo import (
    FIXED_SPEED_STD,
    FIXED_STEERING_STD,
    UPDATE_EPOCHS,
    VALUE_LOSS_WEIGHT,
    train_epoch,
)
from utils import (
    find_opponent_start_index,
    get_ego_idx_range,
    load_racetrack_config,
    load_raceline,
    require_end2race_runtime,
)

ARTIFACT_DIR = Path("checkpoint/ppo")
EVAL_INTERVAL = 1
POLICY_LEARNING_RATE_STEP = 1e-6
POLICY_MAX_LEARNING_RATE = 1e-5
POLICY_LEARNING_RATE_WARMUP_EPOCHS = 10
VALUE_LEARNING_RATE = 1e-5
MAX_EPOCHS = 100
MINIMUM_SAFETY_RATE = 0.9
MINIMUM_OVERTAKE_RATE = 0.6


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run End2Race PPO training and evaluation")

    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoint/epoch_00500.pt"))

    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--ego_raceline", type=str, default="raceline1")
    parser.add_argument("--opponent_racelines", nargs="+", default=["raceline0", "raceline1", "raceline2"])
    parser.add_argument("--opponent_speed_scales", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--num_startpoints", type=int, default=80)
    parser.add_argument("--interval_index", type=int, default=15)
    parser.add_argument("--episode_duration", type=float, default=8.0)
    parser.add_argument("--num_envs", type=int, default=16)

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


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def build_scenarios(settings):
    ego_waypoints = load_raceline(settings.map_name, f"{settings.ego_raceline}.csv")
    ego_indices = get_ego_idx_range(settings.map_name, settings.ego_raceline, settings.num_startpoints)

    scenarios = []
    for opponent_raceline in settings.opponent_racelines:
        if opponent_raceline == settings.ego_raceline:
            opponent_waypoints = ego_waypoints
        else:
            opponent_waypoints = load_raceline(settings.map_name, f"{opponent_raceline}.csv")
        for opponent_speed_scale in settings.opponent_speed_scales:
            for ego_idx in ego_indices:
                opponent_idx = find_opponent_start_index(
                    ego_waypoints,
                    opponent_waypoints,
                    ego_idx,
                    settings.interval_index,
                )
                scenarios.append(
                    Scenario(
                        f"e{ego_idx:04d}_{opponent_raceline}_s{opponent_speed_scale}",
                        int(ego_idx),
                        int(opponent_idx),
                        opponent_raceline,
                        float(opponent_speed_scale),
                    )
                )
    return tuple(scenarios)


def summarize(
    epoch,
    policy_learning_rate,
    value_learning_rate,
    scenarios,
    rollout_summary,
    statistics,
    eval_records,
    saved_checkpoint,
):
    trajectories = rollout_summary["trajectories"]
    records = [trajectory["record"] for trajectory in trajectories]
    transition_count = sum(record["episode_steps"] for record in records)
    return_sum = sum(trajectory["return_sum"] for trajectory in trajectories)
    return_square_sum = sum(
        trajectory["return_square_sum"] for trajectory in trajectories
    )
    error_sum = sum(trajectory["error_sum"] for trajectory in trajectories)
    error_square_sum = sum(
        trajectory["error_square_sum"] for trajectory in trajectories
    )
    return_variance = max(
        return_square_sum / transition_count - (return_sum / transition_count) ** 2,
        0.0,
    )
    error_variance = max(
        error_square_sum / transition_count - (error_sum / transition_count) ** 2,
        0.0,
    )
    evaluated = eval_records is not None
    failures = sum(record["ego_collision"] for record in eval_records) if evaluated else None
    eval_overtakes = (
        sum(record["outcome"] == "overtake" for record in eval_records)
        if evaluated
        else None
    )
    metrics = {
        "epoch": epoch,
        "policy_learning_rate": policy_learning_rate,
        "value_learning_rate": value_learning_rate,
        "evaluated": evaluated,
        "eval_count": len(eval_records) if evaluated else 0,
        "eval_failures": failures,
        "eval_passes": len(eval_records) - failures if evaluated else None,
        "safety": 1.0 - failures / len(eval_records) if evaluated else None,
        "eval_overtakes": eval_overtakes,
        "overtake_rate": eval_overtakes / len(eval_records) if evaluated else None,
        "saved": saved_checkpoint is not None,
        "saved_checkpoint": saved_checkpoint,
        "scenarios": len(scenarios),
        "batches": rollout_summary["batch_count"],
        "trajectories": len(records),
        "transitions": int(transition_count),
        "collisions": sum(record["outcome"] == "ego_collision" for record in records),
        "overtakes": sum(record["outcome"] == "overtake" for record in records),
        "follows": sum(record["outcome"] == "follow" for record in records),
        "mean_steps": float(np.mean([record["episode_steps"] for record in records])) if records else 0.0,
        "mean_return": (
            float(np.mean([trajectory["episode_return"] for trajectory in trajectories]))
            if records
            else 0.0
        ),
        "explained_variance": 1.0 - error_variance / return_variance if return_variance >= 1e-6 else 0.0,
    }
    for name, values in statistics.items():
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_max"] = float(np.max(values))
    return metrics


def resolved_config(args, scenario_count, world_size):
    return {
        **vars(args),
        "checkpoint_path": str(args.checkpoint_path),
        "fixed_steering_std": FIXED_STEERING_STD,
        "fixed_speed_std": FIXED_SPEED_STD,
        "value_weight": VALUE_LOSS_WEIGHT,
        "progress_reward": RaceEnv.PROGRESS_REWARD_WEIGHT,
        "overtake_distance": load_racetrack_config().vehicle.length,
        "collision_penalty": RaceEnv.COLLISION_PENALTY,
        "maximum_ego_speed": RaceEnv.MAXIMUM_EGO_SPEED,
        "batch_size": scenario_count,
        "gpu_processes": world_size,
        "workers_per_gpu": args.num_envs,
        "total_env_workers": args.num_envs * world_size,
        "rollout_batch_size_per_gpu": args.num_envs,
        "trajectories": 1,
        "update_epochs": UPDATE_EPOCHS,
        "policy_learning_rate_start": POLICY_LEARNING_RATE_STEP,
        "policy_learning_rate_step": POLICY_LEARNING_RATE_STEP,
        "policy_learning_rate_cap": POLICY_MAX_LEARNING_RATE,
        "policy_learning_rate_warmup_epochs": POLICY_LEARNING_RATE_WARMUP_EPOCHS,
        "value_learning_rate": VALUE_LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "minimum_safety_rate": MINIMUM_SAFETY_RATE,
        "minimum_overtake_rate": MINIMUM_OVERTAKE_RATE,
        "eval_interval": EVAL_INTERVAL,
    }


def initialize_process_group():
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun PPO requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def synchronize_model(model):
    if not dist.is_initialized():
        return
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


def main():
    args = parse_arguments()
    require_end2race_runtime()
    rank, world_size, device = initialize_process_group()
    envs = None
    try:
        artifact_dir = ARTIFACT_DIR
        config_path = artifact_dir / "config.json"
        checkpoints_path = artifact_dir / "checkpoints.json"
        episodes_path = artifact_dir / "episodes.jsonl"
        metrics_path = artifact_dir / "metrics.jsonl"
        artifacts = (config_path, checkpoints_path, episodes_path, metrics_path)
        artifact_error = None
        if rank == 0:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            existing = [path.name for path in artifacts if path.exists()]
            existing.extend(path.name for path in artifact_dir.glob("ppo*.pt"))
            existing.sort()
            if existing:
                artifact_error = (
                    f"Cannot start PPO in {artifact_dir}: existing "
                    f"{', '.join(existing)}"
                )
        if dist.is_initialized():
            errors = [artifact_error]
            dist.broadcast_object_list(errors, src=0)
            artifact_error = errors[0]
        if artifact_error:
            raise FileExistsError(artifact_error)

        scenarios = build_scenarios(args)
        if world_size > len(scenarios):
            raise ValueError(
                f"Cannot distribute {len(scenarios)} scenarios across "
                f"{world_size} GPU processes"
            )
        rng = np.random.default_rng()
        model = ActorCritic(
            args.checkpoint_path,
            FIXED_STEERING_STD,
            FIXED_SPEED_STD,
        ).to(device)
        synchronize_model(model)
        optimizer = torch.optim.Adam([
            {
                "params": model.model.parameters(),
                "lr": POLICY_LEARNING_RATE_STEP,
                "name": "policy",
            },
            {
                "params": model.value_head.parameters(),
                "lr": VALUE_LEARNING_RATE,
                "name": "value",
            },
        ])
        checkpoint_count = 0
        checkpoint_records = []
        if rank == 0:
            config_path.write_text(
                json.dumps(
                    resolved_config(args, len(scenarios), world_size),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoints_path.write_text("[]\n", encoding="utf-8")
            episodes_path.touch(exist_ok=True)
            metrics_path.touch(exist_ok=True)
            print(
                f"Scenario pool: {len(scenarios)} | GPU processes: {world_size} | "
                f"workers/GPU: {args.num_envs} | "
                f"total workers: {args.num_envs * world_size}"
            )

        envs = VectorEnv(args.num_envs, args)
        for epoch in range(1, MAX_EPOCHS + 1):
            policy_learning_rate = (
                epoch * POLICY_LEARNING_RATE_STEP
                if epoch < POLICY_LEARNING_RATE_WARMUP_EPOCHS
                else POLICY_MAX_LEARNING_RATE
            )
            for parameter_group in optimizer.param_groups:
                if parameter_group["name"] == "policy":
                    parameter_group["lr"] = policy_learning_rate
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
                    append_record(
                        episodes_path,
                        {"epoch": epoch, "phase": "training", **trajectory["record"]},
                    )

            eval_records = evaluate_scenarios(envs, model, scenarios, device, f"Epoch {epoch}")
            if rank == 0:
                for record in eval_records:
                    append_record(episodes_path, {"epoch": epoch, "phase": "screening", **record})

                failures = sum(record["ego_collision"] for record in eval_records)
                safety = 1.0 - failures / len(eval_records)
                eval_overtakes = sum(record["outcome"] == "overtake" for record in eval_records)
                overtake_rate = eval_overtakes / len(eval_records)
                saved_checkpoint = None
                if safety > MINIMUM_SAFETY_RATE and overtake_rate > MINIMUM_OVERTAKE_RATE:
                    checkpoint_count += 1
                    saved_checkpoint = f"ppo_{checkpoint_count:03d}.pt"

                metrics = summarize(
                    epoch,
                    policy_learning_rate,
                    VALUE_LEARNING_RATE,
                    ordered,
                    rollout_summary,
                    statistics,
                    eval_records,
                    saved_checkpoint,
                )
                if saved_checkpoint is not None:
                    torch.save(model.model.state_dict(), artifact_dir / saved_checkpoint)
                    checkpoint_records.append(metrics)
                    checkpoints_path.write_text(
                        json.dumps(checkpoint_records, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"Epoch {epoch} saved {saved_checkpoint}: safety {safety:.2%} | "
                        f"overtake {overtake_rate:.2%}",
                        flush=True,
                    )
                append_record(metrics_path, metrics)
                checkpoint = saved_checkpoint if saved_checkpoint is not None else "none"
                print(
                    f"Epoch {epoch} | safety {metrics['safety']:.2%} | "
                    f"overtake {metrics['overtake_rate']:.2%} | checkpoint {checkpoint} | "
                    f"screened {metrics['eval_count']} | failed {metrics['eval_failures']} | "
                    f"training scenarios {metrics['scenarios']} | collisions {metrics['collisions']} | "
                    f"follow {metrics['follows']} | overtakes {metrics['overtakes']} | "
                    f"return {metrics['mean_return']:.4f}"
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
