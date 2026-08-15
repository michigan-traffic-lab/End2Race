import numpy as np
import torch

ADVANTAGE_EPSILON = 1e-6


def collect_group(vector_env, actor, critic, scenario, device):
    """Roll one scenario across every worker under a frozen policy and independent noise."""
    observations = vector_env.reset_group(scenario)
    actor_hidden = actor.initial_hidden(vector_env.num_envs, device)
    critic_hidden = critic.initial_hidden(vector_env.num_envs, device)
    active = [True] * vector_env.num_envs
    trajectories = [{"observations": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "bootstrap": 0.0} for _ in range(vector_env.num_envs)]
    records = [None] * vector_env.num_envs

    while any(active):
        with torch.no_grad():
            observation_batch = torch.as_tensor(observations, device=device)
            actions, log_probs, next_actor_hidden = actor.act(observation_batch, actor_hidden)
            values, next_critic_hidden = critic.step_values(observation_batch, critic_hidden)
        actions = actions.cpu().numpy()
        log_probs = log_probs.cpu().numpy()
        values = values.cpu().numpy()

        bootstrap_ranks = []
        for rank, result in enumerate(vector_env.step(actions, active)):
            if result is None:
                continue
            next_observation, reward, done, info = result
            trajectory = trajectories[rank]
            trajectory["observations"].append(observations[rank].copy())
            trajectory["actions"].append(actions[rank])
            trajectory["log_probs"].append(log_probs[rank])
            trajectory["values"].append(values[rank])
            trajectory["rewards"].append(reward)
            observations[rank] = next_observation
            if done:
                active[rank] = False
                records[rank] = info
                # Only an ego collision is a true terminal state; every other ending is cut short
                if not info["ego_collision"]:
                    bootstrap_ranks.append(rank)

        if bootstrap_ranks:
            with torch.no_grad():
                tail_observations = torch.as_tensor(observations[bootstrap_ranks], device=device)
                tail_hidden = next_critic_hidden[:, bootstrap_ranks].contiguous()
                tail_values = critic.step_values(tail_observations, tail_hidden)[0].cpu().numpy()
            for slot, rank in enumerate(bootstrap_ranks):
                trajectories[rank]["bootstrap"] = float(tail_values[slot])

        actor_hidden = next_actor_hidden
        critic_hidden = next_critic_hidden

    group = []
    for trajectory, record in zip(trajectories, records):
        group.append({
            "observations": np.asarray(trajectory["observations"], dtype=np.float32),
            "actions": np.asarray(trajectory["actions"], dtype=np.float32),
            "log_probs": np.asarray(trajectory["log_probs"], dtype=np.float32),
            "values": np.asarray(trajectory["values"], dtype=np.float32),
            "rewards": np.asarray(trajectory["rewards"], dtype=np.float32),
            "bootstrap": trajectory["bootstrap"],
            "record": record,
        })
    # cuDNN evaluates a 1-step GRU call differently from a full-sequence one, so restate pi_old on the update path
    _refresh_old_estimates(group, actor, critic, device)
    return group


def _refresh_old_estimates(group, actor, critic, device):
    """Recompute log-probabilities and values on the batched replay path the update uses."""
    lengths = [len(trajectory["rewards"]) for trajectory in group]
    observations = np.zeros((len(group), max(lengths), group[0]["observations"].shape[1]), dtype=np.float32)
    actions = np.zeros((len(group), max(lengths), 2), dtype=np.float32)
    for slot, trajectory in enumerate(group):
        observations[slot, : lengths[slot]] = trajectory["observations"]
        actions[slot, : lengths[slot]] = trajectory["actions"]

    with torch.no_grad():
        observations = torch.as_tensor(observations, device=device)
        log_probs = actor.evaluate(observations, torch.as_tensor(actions, device=device)).cpu().numpy()
        values = critic.evaluate(observations).cpu().numpy()
    for slot, trajectory in enumerate(group):
        trajectory["log_probs"] = log_probs[slot, : lengths[slot]].astype(np.float32)
        trajectory["values"] = values[slot, : lengths[slot]].astype(np.float32)


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


def score_group(group, gamma, gae_lambda, advantage_mode, minimum_spread):
    """Attach per-step advantages and returns, then report the group return spread."""
    for trajectory in group:
        generalized_advantages(trajectory, gamma, gae_lambda)
        trajectory["episode_return"] = discounted_return(trajectory["rewards"], gamma)

    returns = np.asarray([trajectory["episode_return"] for trajectory in group], dtype=np.float64)
    spread = float(returns.std())
    if advantage_mode == "group":
        # Below the measured collision-free spread band a group carries noise, not signal
        if spread < minimum_spread:
            relative = np.zeros_like(returns)
        else:
            relative = (returns - returns.mean()) / spread
        for trajectory, advantage in zip(group, relative):
            trajectory["advantages"] = np.full_like(trajectory["advantages"], float(advantage))
    return spread


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


def train_networks(actor, critic, actor_optimizer, critic_optimizer, groups, args, rng, device, epochs, update_actor):
    """Run the clipped surrogate and the value regression over shuffled minibatches of whole groups."""
    statistics = {name: [] for name in ("policy_loss", "value_loss", "clip_fraction", "approx_kl", "actor_grad_norm", "critic_grad_norm")}

    for _epoch in range(epochs):
        order = rng.permutation(len(groups))
        for start in range(0, len(order), args.groups_per_minibatch):
            selected = order[start : start + args.groups_per_minibatch]
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
            if update_actor:
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

            if args.target_kl is not None and statistics["approx_kl"][-1] > args.target_kl:
                return statistics

    return statistics


def explained_variance(groups):
    """Report how much of the return variance the rollout critic already predicted."""
    values = np.concatenate([trajectory["values"] for group in groups for trajectory in group])
    returns = np.concatenate([trajectory["returns"] for group in groups for trajectory in group])
    variance = returns.var()
    if variance < ADVANTAGE_EPSILON:
        return 0.0
    return float(1.0 - (returns - values).var() / variance)
