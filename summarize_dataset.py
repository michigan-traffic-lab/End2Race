#!/usr/bin/env python3
"""Summarize an existing End2Race collection without running simulation."""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import load_project_config


ARTIFACT_PATTERN = re.compile(
    r"^(?P<state>[fo])_ol(?P<opponent_raceline>[^_]+)"
    r"_e(?P<ego_idx>\d+)_i(?P<interval_idx>\d+)"
    r"_o(?P<opponent_idx>\d+)_s(?P<speed_scale>\d+(?:\.\d+)?)$"
)


def parse_bool(value):
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create summary.json from existing collection artifacts."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--ego-raceline", required=True)
    parser.add_argument("--num-startpoints", required=True, type=int)
    parser.add_argument("--sim-duration", required=True, type=float)
    parser.add_argument("--sample-interval", required=True, type=float)
    parser.add_argument("--render", required=True, type=parse_bool)
    parser.add_argument("--interval-index", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument(
        "--opponent-racelines", required=True, nargs="+"
    )
    parser.add_argument(
        "--opponent-speed-scales", required=True, nargs="+", type=float
    )
    parser.add_argument("--collection-failures", type=int, default=0)
    return parser.parse_args()


def parse_artifact(path, outcome, raceline_names):
    match = ARTIFACT_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected collection artifact name: {path}")

    values = match.groupdict()
    raceline_token = values["opponent_raceline"]
    opponent_raceline = raceline_names.get(
        raceline_token, f"raceline{raceline_token}"
    )
    return {
        "outcome": outcome,
        "final_state": (
            "overtaking" if values["state"] == "o" else "following"
        ),
        "opponent_raceline": opponent_raceline,
        "ego_idx": int(values["ego_idx"]),
        "interval_idx": int(values["interval_idx"]),
        "opponent_idx": int(values["opponent_idx"]),
        "speed_scale": float(values["speed_scale"]),
        "path": path,
    }


def count_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def summarize(args):
    dataset_dir = args.dataset_dir
    success_paths = sorted((dataset_dir / "success").glob("*.csv"))
    collision_paths = sorted((dataset_dir / "collision").glob("*.json"))

    raceline_names = {
        name.replace("raceline", "").replace(".csv", ""): name
        for name in args.opponent_racelines
    }
    episodes = [
        parse_artifact(path, "collision_free", raceline_names)
        for path in success_paths
    ] + [
        parse_artifact(path, "collision", raceline_names)
        for path in collision_paths
    ]

    episode_keys = [
        (
            episode["opponent_raceline"],
            episode["ego_idx"],
            episode["interval_idx"],
            episode["speed_scale"],
        )
        for episode in episodes
    ]
    duplicate_keys = [
        key for key, count in Counter(episode_keys).items() if count > 1
    ]
    if duplicate_keys:
        raise ValueError(
            "Duplicate scenario artifacts found; clear stale results before "
            f"summarizing. First duplicate: {duplicate_keys[0]}"
        )

    expected_racelines = set(args.opponent_racelines)
    expected_speed_scales = set(args.opponent_speed_scales)
    for episode in episodes:
        if episode["opponent_raceline"] not in expected_racelines:
            raise ValueError(
                "Artifact uses unexpected opponent raceline: "
                f"{episode['path']}"
            )
        if episode["speed_scale"] not in expected_speed_scales:
            raise ValueError(
                "Artifact uses unexpected opponent speed scale: "
                f"{episode['path']}"
            )
        if episode["interval_idx"] != args.interval_index:
            raise ValueError(
                f"Artifact uses unexpected interval index: {episode['path']}"
            )

    project_config = load_project_config()
    expected_scenarios = (
        args.num_startpoints
        * len(args.opponent_racelines)
        * len(args.opponent_speed_scales)
    )
    recorded_scenarios = len(episodes)
    collision_free = sum(
        episode["outcome"] == "collision_free" for episode in episodes
    )
    collisions = sum(
        episode["outcome"] == "collision" for episode in episodes
    )
    successful_overtakes = sum(
        episode["outcome"] == "collision_free"
        and episode["final_state"] == "overtaking"
        for episode in episodes
    )
    collision_free_following = sum(
        episode["outcome"] == "collision_free"
        and episode["final_state"] == "following"
        for episode in episodes
    )
    collision_final_states = Counter(
        episode["final_state"]
        for episode in episodes
        if episode["outcome"] == "collision"
    )
    ego_indices = sorted({episode["ego_idx"] for episode in episodes})
    training_rows = sum(count_csv_rows(path) for path in success_paths)

    breakdown = []
    for raceline in args.opponent_racelines:
        for speed_scale in args.opponent_speed_scales:
            subset = [
                episode
                for episode in episodes
                if episode["opponent_raceline"] == raceline
                and episode["speed_scale"] == speed_scale
            ]
            subset_collision_free = sum(
                episode["outcome"] == "collision_free"
                for episode in subset
            )
            subset_collisions = sum(
                episode["outcome"] == "collision" for episode in subset
            )
            subset_overtakes = sum(
                episode["outcome"] == "collision_free"
                and episode["final_state"] == "overtaking"
                for episode in subset
            )
            breakdown.append(
                {
                    "opponent_raceline": raceline,
                    "opponent_speed_scale": speed_scale,
                    "recorded_scenarios": len(subset),
                    "collision_free_scenarios": subset_collision_free,
                    "collision_scenarios": subset_collisions,
                    "successful_overtakes": subset_overtakes,
                    "collision_free_following": (
                        subset_collision_free - subset_overtakes
                    ),
                    "collision_free_rate_percent": percentage(
                        subset_collision_free, len(subset)
                    ),
                    "successful_overtake_rate_percent": percentage(
                        subset_overtakes, len(subset)
                    ),
                }
            )

    success_videos = len(list((dataset_dir / "success").glob("*.mp4")))
    collision_videos = len(
        list((dataset_dir / "collision").glob("*.mp4"))
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source": "existing_collection_artifacts",
        "collection_config": {
            "mode": "multi_agent",
            "map_name": args.map_name,
            "ego_raceline": args.ego_raceline,
            "num_startpoints": args.num_startpoints,
            "ego_indices": ego_indices,
            "opponent_racelines": args.opponent_racelines,
            "opponent_speed_scales": args.opponent_speed_scales,
            "interval_index": args.interval_index,
            "simulation_duration_seconds": args.sim_duration,
            "sample_interval_seconds": args.sample_interval,
            "render": args.render,
            "workers": args.workers,
        },
        "data_config": {
            "lidar_features": project_config.model.lidar_features,
            "csv_columns": 4 + project_config.model.lidar_features,
            "vehicle": vars(project_config.vehicle),
            "expert": vars(project_config.expert),
        },
        "results": {
            "expected_scenarios": expected_scenarios,
            "recorded_scenarios": recorded_scenarios,
            "missing_scenarios": max(
                expected_scenarios - recorded_scenarios, 0
            ),
            "unexpected_extra_scenarios": max(
                recorded_scenarios - expected_scenarios, 0
            ),
            "collection_process_failures": args.collection_failures,
            "collision_free_scenarios": collision_free,
            "collision_scenarios": collisions,
            "successful_overtakes": successful_overtakes,
            "collision_free_following": collision_free_following,
            "collision_final_state": {
                "overtaking": collision_final_states["overtaking"],
                "following": collision_final_states["following"],
            },
            "rates_percent_of_recorded": {
                "collision_free": percentage(
                    collision_free, recorded_scenarios
                ),
                "collision": percentage(collisions, recorded_scenarios),
                "successful_overtake": percentage(
                    successful_overtakes, recorded_scenarios
                ),
                "collision_free_following": percentage(
                    collision_free_following, recorded_scenarios
                ),
            },
            "training_rows": training_rows,
            "artifacts": {
                "success_csv": len(success_paths),
                "collision_json": len(collision_paths),
                "success_video": success_videos,
                "collision_video": collision_videos,
            },
            "breakdown": breakdown,
        },
    }
    return summary


def main():
    args = parse_arguments()
    summary = summarize(args)
    output_path = args.dataset_dir / "summary.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    print(f"Dataset summary saved to {output_path}")


if __name__ == "__main__":
    main()
