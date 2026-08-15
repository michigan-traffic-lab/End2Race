def wrapped_progress_delta(current_progress, previous_progress, track_length):
    """Measure signed progress across the cyclic lap boundary."""
    offset = current_progress - previous_progress + 0.5 * track_length
    return float(offset % track_length - 0.5 * track_length)


def transition_reward(ego_delta, opponent_delta, ego_collision, reward_config):
    """Combine ego progress, progress relative to the opponent, and the collision penalty."""
    reward = reward_config.progress_weight * ego_delta
    reward += reward_config.relative_weight * (ego_delta - opponent_delta)
    if ego_collision:
        reward += reward_config.collision_penalty
    return float(reward)
