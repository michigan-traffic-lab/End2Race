import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ppo.env import RaceEnv, Scenario, VectorEnv
from eval_ppo import evaluate_scenarios
from eval_single import evaluate_laps
from ppo.policy import ActorCritic
from train_ppo import (
    INITIAL_SPEED_STD,
    INITIAL_STEERING_STD,
    UPDATE_EPOCHS,
    VALUE_LOSS_WEIGHT,
    explained_variance,
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
EVAL_MAPS = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
EVAL_INTERVAL = 5


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
    parser.add_argument("--num_envs", type=int, default=48)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.1)

    args = parser.parse_args()
    positive_values = {
        "--num_startpoints": args.num_startpoints,
        "--episode_duration": args.episode_duration,
        "--num_envs": args.num_envs,
        "--learning_rate": args.learning_rate,
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


def evaluate_single(model, device, vehicle, epoch):
    results = {}
    model.model.eval()
    try:
        for map_name in EVAL_MAPS:
            print(f"Epoch {epoch} single vehicle: {map_name} running", flush=True)
            settings = SimpleNamespace(
                map_name=map_name,
                noise=0.0,
                seed=None,
                lap_num=1,
                start_idx=0,
                minimum_lap_time=10.0,
                render=False,
            )
            with redirect_stdout(io.StringIO()):
                result = evaluate_laps(model.model, device, vehicle, settings)
            results[map_name] = result
            status = "passed" if result["passed"] else "failed"
            print(
                f"Epoch {epoch} single vehicle: {map_name} {status} | "
                f"laps {result['laps_completed']} | progress {result['lap_progress']:.3f} | "
                f"time {result['lap_time']:.3f}s",
                flush=True,
            )
            if not result["passed"]:
                return results, False
    finally:
        model.train()
    return results, True


def summarize(
    epoch,
    scenarios,
    batches,
    statistics,
    action_std,
    eval_records,
    single_results,
    single_passed,
    best_safety,
    saved,
):
    records = [trajectory["record"] for batch in batches for trajectory in batch]
    evaluated = eval_records is not None
    failures = sum(record["ego_collision"] for record in eval_records) if evaluated else None
    metrics = {
        "epoch": epoch,
        "evaluated": evaluated,
        "eval_count": len(eval_records) if evaluated else 0,
        "eval_failures": failures,
        "eval_passes": len(eval_records) - failures if evaluated else None,
        "safety": 1.0 - failures / len(eval_records) if evaluated else None,
        "single_passed": single_passed,
        "single": single_results,
        "best_safety": best_safety,
        "saved": saved,
        "scenarios": len(scenarios),
        "batches": len(batches),
        "steering_std": action_std[0],
        "speed_std": action_std[1],
        "trajectories": len(records),
        "transitions": int(sum(record["episode_steps"] for record in records)),
        "collisions": sum(record["outcome"] == "ego_collision" for record in records),
        "overtakes": sum(record["outcome"] == "overtake" for record in records),
        "follows": sum(record["outcome"] == "follow" for record in records),
        "mean_steps": float(np.mean([record["episode_steps"] for record in records])) if records else 0.0,
        "mean_return": float(np.mean([trajectory["episode_return"] for batch in batches for trajectory in batch])) if records else 0.0,
        "explained_variance": explained_variance(batches) if batches else 0.0,
    }
    for name, values in statistics.items():
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_max"] = float(np.max(values))
    return metrics


def resolved_config(args):
    return {
        **vars(args),
        "checkpoint_path": str(args.checkpoint_path),
        "steering_std": INITIAL_STEERING_STD,
        "speed_std": INITIAL_SPEED_STD,
        "value_weight": VALUE_LOSS_WEIGHT,
        "progress_reward": RaceEnv.PROGRESS_REWARD_WEIGHT,
        "overtake_distance": load_racetrack_config().vehicle.length,
        "collision_penalty": RaceEnv.COLLISION_PENALTY,
        "batch_size": args.num_envs,
        "trajectories": 1,
        "update_epochs": UPDATE_EPOCHS,
        "eval_interval": EVAL_INTERVAL,
        "eval_maps": list(EVAL_MAPS),
    }


def main():
    args = parse_arguments()
    require_end2race_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifact_dir = ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ppo_path = artifact_dir / "ppo.pt"
    config_path = artifact_dir / "config.json"
    episodes_path = artifact_dir / "episodes.jsonl"
    metrics_path = artifact_dir / "metrics.jsonl"
    artifacts = (ppo_path, config_path, episodes_path, metrics_path)
    existing = [path.name for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(f"Cannot start PPO in {artifact_dir}: existing {', '.join(existing)}")

    scenarios = build_scenarios(args)
    vehicle = load_racetrack_config().vehicle
    rng = np.random.default_rng()
    model = ActorCritic(args.checkpoint_path, INITIAL_STEERING_STD, INITIAL_SPEED_STD).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_safety = None
    config_path.write_text(json.dumps(resolved_config(args), indent=2) + "\n", encoding="utf-8")
    episodes_path.touch(exist_ok=True)
    metrics_path.touch(exist_ok=True)

    print(f"Scenario pool: {len(scenarios)} | workers: {args.num_envs} | epochs: unbounded | eval every {EVAL_INTERVAL} | device: {device}")
    envs = VectorEnv(args.num_envs, args)
    try:
        epoch = 1
        while True:
            ordered, batches, statistics = train_epoch(
                model,
                optimizer,
                envs,
                scenarios,
                rng,
                args,
                device,
                epoch,
            )
            for batch in batches:
                for trajectory in batch:
                    append_record(episodes_path, {"epoch": epoch, "phase": "training", **trajectory["record"]})

            eval_records = None
            single_results = None
            single_passed = None
            saved = False
            if epoch % EVAL_INTERVAL == 0:
                eval_records = evaluate_scenarios(envs, model, scenarios, device, f"Epoch {epoch}")
                for record in eval_records:
                    append_record(episodes_path, {"epoch": epoch, "phase": "screening", **record})

                failures = sum(record["ego_collision"] for record in eval_records)
                safety = 1.0 - failures / len(eval_records)
                single_results, single_passed = evaluate_single(model, device, vehicle, epoch)
                saved = single_passed and (best_safety is None or safety > best_safety)
                if saved:
                    best_safety = safety
                    torch.save(model.model.state_dict(), ppo_path)
                    print(f"Epoch {epoch} best checkpoint: safety {safety:.2%} | single vehicle passed", flush=True)
                elif not single_passed:
                    failed_map = next(map_name for map_name, result in single_results.items() if not result["passed"])
                    print(f"Epoch {epoch} checkpoint skipped: single vehicle failed on {failed_map}", flush=True)

            action_std = model.action_std.detach().cpu().tolist()
            metrics = summarize(
                epoch,
                ordered,
                batches,
                statistics,
                action_std,
                eval_records,
                single_results,
                single_passed,
                best_safety,
                saved,
            )
            append_record(metrics_path, metrics)
            if metrics["evaluated"]:
                best = f"{best_safety:.2%}" if best_safety is not None else "none"
                print(
                    f"Epoch {epoch} | safety {metrics['safety']:.2%} | best {best} | "
                    f"single vehicle {'passed' if single_passed else 'failed'} | "
                    f"screened {metrics['eval_count']} | failed {metrics['eval_failures']} | "
                    f"training scenarios {metrics['scenarios']} | collisions {metrics['collisions']} | "
                    f"follow {metrics['follows']} | overtakes {metrics['overtakes']} | "
                    f"return {metrics['mean_return']:.4f} | std ({action_std[0]:.3f}, {action_std[1]:.3f})"
                )
            else:
                print(
                    f"Epoch {epoch} | training scenarios {metrics['scenarios']} | batches {metrics['batches']} | "
                    f"collisions {metrics['collisions']} | follow {metrics['follows']} | "
                    f"overtakes {metrics['overtakes']} | return {metrics['mean_return']:.4f} | "
                    f"std ({action_std[0]:.3f}, {action_std[1]:.3f})"
                )
            epoch += 1
    finally:
        envs.close()


if __name__ == "__main__":
    main()
