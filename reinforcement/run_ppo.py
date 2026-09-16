import itertools
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist

import yaml
from reinforcement.env import Scenario, VectorEnv
from reinforcement.eval_ppo import evaluate_scenarios
from reinforcement.policy import ActorCritic
from reinforcement.train_ppo import train_epoch
from expert.utils import find_opponent_start_index, get_ego_idx_range
from f1tenth_sim.utils import load_raceline

ROOT = Path(__file__).resolve().parents[1]


def _append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _build_scenarios(settings):
    scenarios = []
    for map_name in settings['maps']:
        ego_waypoints = load_raceline(map_name, f"{settings['ego_raceline']}.csv")
        ego_indices = get_ego_idx_range(map_name, settings['ego_raceline'], settings['num_startpoints'])

        for opponent_raceline in settings['opponent_racelines']:
            if opponent_raceline == settings['ego_raceline']:
                opponent_waypoints = ego_waypoints
            else:
                opponent_waypoints = load_raceline(map_name, f"{opponent_raceline}.csv")
            for opponent_speed_scale in settings['opponent_speed_scales']:
                for ego_idx in ego_indices:
                    opponent_idx = find_opponent_start_index(
                        ego_waypoints,
                        opponent_waypoints,
                        ego_idx,
                        settings['interval_index'],
                    )
                    scenarios.append(
                        Scenario(
                            f"{map_name}_e{ego_idx:04d}_{opponent_raceline}_s{opponent_speed_scale}",
                            map_name,
                            int(ego_idx),
                            int(opponent_idx),
                            opponent_raceline,
                            float(opponent_speed_scale),
                        )
                    )
    return tuple(scenarios)


def _summarize(
    config,
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
        "learning_rate": config['ppo']['learning_rate'],
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
        ),
        "training_mean_return": (
            float(np.mean([trajectory["episode_return"] for trajectory in trajectories]))
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


def _initialize_process_group(config):
    device = torch.device(config['runtime']['device'])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA.")
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{(device.index or 0) + local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
        return dist.get_rank(), dist.get_world_size(), device
    if device.index is not None:
        torch.cuda.set_device(device)
    return 0, 1, device


def _synchronize_model(model):
    if not dist.is_initialized():
        return
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


def main():
    with (ROOT / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    rank, world_size, device = _initialize_process_group(config)
    envs = None
    try:
        artifact_dir = ROOT / config['ppo']['output_dir']
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

        scenarios = _build_scenarios(config["ppo_environment"])
        if world_size > len(scenarios):
            raise ValueError(
                f"Cannot distribute {len(scenarios)} scenarios across "
                f"{world_size} GPU processes"
            )
        model = ActorCritic(
            ROOT / config['ppo']['initial_policy_path'],
            config['ppo_exploration']['initial_steering_std'],
            config['ppo_exploration']['initial_speed_std'],
        ).to(device)
        _synchronize_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['ppo']['learning_rate'])

        if rank == 0:
            config_path.write_text(
                json.dumps(
                    config,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            episodes_path.touch(exist_ok=True)
            metrics_path.touch(exist_ok=True)
            print(
                f"Scenario pool: {len(scenarios)} | GPU processes: {world_size} | "
                f"workers/GPU: {config['runtime']['workers']} | "
                f"total workers: {config['runtime']['workers'] * world_size} | "
                f"lr: {config['ppo']['learning_rate']:.1e}"
            )

        envs = VectorEnv(config['runtime']['workers'], config)
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
                config,
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
            model.action_std.mul_(config['ppo_exploration']['action_std_decay'])
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
                    config,
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
                    metrics["screening_safety_rate"] > config['ppo_screening']['minimum_safety_rate']
                    and metrics["screening_overtake_rate"] > config['ppo_screening']['minimum_overtake_rate']
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
