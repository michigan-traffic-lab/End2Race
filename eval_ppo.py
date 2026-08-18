import time

import torch
import torch.distributed as dist

from ppo.env import shard_scenarios


def _evaluate_group(envs, model, scenarios, device):
    observations = envs.reset_scenarios(scenarios)
    hidden = model.initial_hidden(len(scenarios), device)
    active = [True] * len(scenarios)
    records = [None] * len(scenarios)

    while any(active):
        with torch.no_grad():
            observation_batch = torch.as_tensor(observations, device=device)
            actions, next_hidden = model.predict(observation_batch, hidden)
        actions = actions.cpu().numpy()

        for slot, result in enumerate(envs.step(actions, active)):
            if result is None:
                continue
            next_observation, _, done, info = result
            observations[slot] = next_observation
            if done:
                active[slot] = False
                records[slot] = info
        hidden = next_hidden
    return records


def _evaluate_shard(envs, model, scenarios, device, label, started_at):
    records = []
    print(f"{label} screening: 0/{len(scenarios)}", flush=True)
    for start in range(0, len(scenarios), envs.num_envs):
        selected = scenarios[start : start + envs.num_envs]
        records.extend(_evaluate_group(envs, model, selected, device))
        failures = sum(record["ego_collision"] for record in records)
        elapsed = time.monotonic() - started_at
        print(
            f"{label} screening: {len(records)}/{len(scenarios)} | "
            f"failed {failures} | elapsed {elapsed:.0f}s",
            flush=True,
        )
    return records


def evaluate_scenarios(envs, model, scenarios, device, label):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    shard = shard_scenarios(scenarios, world_size)[rank]
    started_at = time.monotonic()
    if rank == 0:
        print(f"{label} screening: 0/{len(scenarios)}", flush=True)
    local_records = _evaluate_shard(
        envs,
        model,
        shard,
        device,
        label,
        started_at,
    )
    if dist.is_initialized():
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_records, gathered, dst=0)
        if gathered is None:
            return None
        records = [record for rank_records in gathered for record in rank_records]
    else:
        records = local_records
    failures = sum(record["ego_collision"] for record in records)
    elapsed = time.monotonic() - started_at
    print(
        f"{label} screening complete: {len(records)}/{len(scenarios)} | "
        f"failed {failures} | elapsed {elapsed:.0f}s",
        flush=True,
    )
    return records
