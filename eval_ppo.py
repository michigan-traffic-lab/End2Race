import time

import torch


def evaluate_scenarios(envs, model, scenarios, device, label):
    records = []
    started_at = time.monotonic()
    print(f"{label} screening: 0/{len(scenarios)}", flush=True)
    for start in range(0, len(scenarios), envs.num_envs):
        selected = scenarios[start : start + envs.num_envs]
        observations = envs.reset_scenarios(selected)
        hidden = model.initial_hidden(len(selected), device)
        active = [True] * len(selected)
        batch_records = [None] * len(selected)

        while any(active):
            with torch.no_grad():
                observation_batch = torch.as_tensor(observations, device=device)
                actions, next_hidden = model.predict(observation_batch, hidden)
            actions = actions.cpu().numpy()

            for rank, result in enumerate(envs.step(actions, active)):
                if result is None:
                    continue
                next_observation, _, done, info = result
                observations[rank] = next_observation
                if done:
                    active[rank] = False
                    batch_records[rank] = info
            hidden = next_hidden

        records.extend(batch_records)
        failures = sum(record["ego_collision"] for record in records)
        elapsed = time.monotonic() - started_at
        print(
            f"{label} screening: {len(records)}/{len(scenarios)} | "
            f"failed {failures} | elapsed {elapsed:.0f}s",
            flush=True,
        )
    return records
