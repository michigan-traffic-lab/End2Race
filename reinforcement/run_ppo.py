import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .env import RaceEnv, Scenario, VectorEnv
from .eval_ppo import evaluate_scenarios
from .policy import ActorCritic
from .train_ppo import (
    ACTION_STD_DECAY,
    INITIAL_SPEED_STD,
    INITIAL_STEERING_STD,
    UPDATE_EPOCHS,
    VALUE_LOSS_WEIGHT,
    train_epoch,
)
from expert.utils import (
    find_opponent_start_index,
    get_ego_idx_range,
    require_end2race_runtime,
)
from f1tenth_sim.utils import load_racetrack_config, load_raceline

ARTIFACT_DIR = Path("checkpoint/ppo")
LEARNING_RATE = 1e-5
MINIMUM_SAFETY_RATE = 0.95
MINIMUM_OVERTAKE_RATE = 0.9


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


def _append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _build_scenarios(settings):
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


def _summarize(
    epoch,
    scenarios,
    rollout_summary,
    statistics,
    eval_records,
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
    screening_count = len(eval_records)
    screening_collision_count = sum(record["ego_collision"] for record in eval_records)
    screening_safe_count = screening_count - screening_collision_count
    screening_overtake_count = sum(
        record["outcome"] == "overtake" for record in eval_records
    )
    metrics = {
        "epoch": epoch,
        "learning_rate": LEARNING_RATE,
        "screening_count": screening_count,
        "screening_safe_count": screening_safe_count,
        "screening_collision_count": screening_collision_count,
        "screening_overtake_count": screening_overtake_count,
        "screening_safety_rate": screening_safe_count / screening_count,
        "screening_collision_rate": screening_collision_count / screening_count,
        "screening_overtake_rate": screening_overtake_count / screening_count,
        "training_scenario_count": len(scenarios),
        "training_batch_count": rollout_summary["batch_count"],
        "training_trajectory_count": len(records),
        "training_transition_count": int(transition_count),
        "training_collision_count": sum(
            record["outcome"] == "ego_collision" for record in records
        ),
        "training_overtake_count": sum(
            record["outcome"] == "overtake" for record in records
        ),
        "training_follow_count": sum(
            record["outcome"] == "follow" for record in records
        ),
        "training_mean_steps": (
            float(np.mean([record["episode_steps"] for record in records]))
            if records
            else 0.0
        ),
        "training_mean_return": (
            float(np.mean([trajectory["episode_return"] for trajectory in trajectories]))
            if records
            else 0.0
        ),
        "training_explained_variance": (
            1.0 - error_variance / return_variance
            if return_variance >= 1e-6
            else 0.0
        ),
    }
    for name, values in statistics.items():
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_max"] = float(np.max(values))
    return metrics


def _resolved_config(args, scenario_count, world_size):
    return {
        **vars(args),
        "checkpoint_path": str(args.checkpoint_path),
        "initial_steering_std": INITIAL_STEERING_STD,
        "initial_speed_std": INITIAL_SPEED_STD,
        "action_std_decay": ACTION_STD_DECAY,
        "value_weight": VALUE_LOSS_WEIGHT,
        "progress_reward": RaceEnv.PROGRESS_REWARD_WEIGHT,
        "overtake_distance": load_racetrack_config().vehicle.length,
        "collision_penalty": RaceEnv.COLLISION_PENALTY,
        "maximum_ego_speed": RaceEnv.MAXIMUM_EGO_SPEED,
        "batch_size": scenario_count,
        "gpu_processes": world_size,
        "workers_per_gpu": args.num_envs,
        "total_env_workers": args.num_envs * world_size,
        "trajectories_per_scenario": 1,
        "update_epochs": UPDATE_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "training_duration": "until_stopped",
        "minimum_safety_rate": MINIMUM_SAFETY_RATE,
        "minimum_overtake_rate": MINIMUM_OVERTAKE_RATE,
    }


def _initialize_process_group():
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun PPO requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def _synchronize_model(model):
    if not dist.is_initialized():
        return
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


def main():
    args = parse_arguments()
    require_end2race_runtime()
    rank, world_size, device = _initialize_process_group()
    envs = None
    try:
        artifact_dir = ARTIFACT_DIR
        config_path = artifact_dir / "config.json"
        episodes_path = artifact_dir / "episodes.jsonl"
        metrics_path = artifact_dir / "metrics.jsonl"
        artifact_error = None
        if rank == 0:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(path.name for path in artifact_dir.iterdir())
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

        scenarios = _build_scenarios(args)
        if world_size > len(scenarios):
            raise ValueError(
                f"Cannot distribute {len(scenarios)} scenarios across "
                f"{world_size} GPU processes"
            )
        model = ActorCritic(
            args.checkpoint_path,
            INITIAL_STEERING_STD,
            INITIAL_SPEED_STD,
        ).to(device)
        _synchronize_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if rank == 0:
            config_path.write_text(
                json.dumps(
                    _resolved_config(
                        args,
                        len(scenarios),
                        world_size,
                    ),
                    indent=2,
                )
                + "\n",
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
                        {
                            "epoch": epoch,
                            "phase": "training",
                            **trajectory["record"],
                        },
                    )

            label = f"Epoch {epoch}"
            eval_records = evaluate_scenarios(envs, model, scenarios, device, label)
            model.action_std.mul_(ACTION_STD_DECAY)
            next_action_std = model.action_std.detach().cpu().tolist()
            if rank == 0:
                for record in eval_records:
                    _append_record(
                        episodes_path,
                        {
                            "epoch": epoch,
                            "phase": "screening",
                            **record,
                        },
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
                    model_path = artifact_dir / f"ppo_{saved_model_count:03d}.pt"
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
