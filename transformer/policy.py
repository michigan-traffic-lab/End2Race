import torch
import torch.nn as nn

from transformer.model import End2RaceTransformer


class TransformerActorCritic(nn.Module):
    """Transformer policy and value head initialized from BC policy weights."""

    def __init__(self, checkpoint_path, steering_std, speed_std):
        super().__init__()
        self.model = End2RaceTransformer()
        self.value_head = nn.Sequential(
            nn.Linear(End2RaceTransformer.MODEL_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.register_buffer(
            "action_std",
            torch.tensor([steering_std, speed_std], dtype=torch.float32),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint)

    def initial_state(self, batch_size, device):
        return None

    @staticmethod
    def select_state(state, indices):
        if state is None:
            return None
        return state[indices].contiguous()

    def _step_outputs(self, observations, context=None):
        lidar = observations[..., :End2RaceTransformer.NUM_LIDAR_FEATURES]
        speed = observations[..., End2RaceTransformer.NUM_LIDAR_FEATURES:]
        features, context = self.model.encode(lidar, speed, context=context)
        return self.model.output_layer(features), self.value_head(features)[..., 0], context

    def _sequence_outputs(self, observations):
        _, sequence_length, _ = observations.shape
        context_length = End2RaceTransformer.CONTEXT_LENGTH
        lengths = torch.arange(1, sequence_length + 1, device=observations.device)
        lengths = lengths.clamp(max=context_length)
        positions = torch.arange(context_length, device=observations.device)[None]
        padding_mask = positions >= lengths[:, None]

        episode_actions = []
        episode_values = []
        for episode in observations:
            windows = observations.new_zeros(
                sequence_length,
                context_length,
                observations.shape[-1],
            )
            for end in range(1, sequence_length + 1):
                start = max(0, end - context_length)
                length = end - start
                windows[end - 1, :length] = episode[start:end]

            lidar = windows[..., :End2RaceTransformer.NUM_LIDAR_FEATURES]
            speed = windows[..., End2RaceTransformer.NUM_LIDAR_FEATURES:]
            features, _ = self.model.encode(lidar, speed, padding_mask=padding_mask)
            batch_indices = torch.arange(sequence_length, device=observations.device)
            final_features = features[batch_indices, lengths - 1]
            episode_actions.append(self.model.output_layer(final_features))
            episode_values.append(self.value_head(final_features)[:, 0])

        return torch.stack(episode_actions), torch.stack(episode_values)

    def _log_prob(self, mean, actions):
        return torch.distributions.Normal(mean, self.action_std).log_prob(actions).sum(-1)

    def act(self, observations, state):
        mean, values, state = self._step_outputs(observations[:, None], state)
        mean = mean[:, 0]
        actions = torch.normal(mean, self.action_std.expand_as(mean))
        return actions, mean, self._log_prob(mean, actions), values[:, 0], state

    def predict(self, observations, state):
        mean, _, state = self._step_outputs(observations[:, None], state)
        return mean[:, 0], state

    def evaluate(self, observations, actions=None, hidden=None):
        if hidden is not None or observations.shape[1] == 1:
            mean, values, state = self._step_outputs(observations, hidden)
        else:
            mean, values = self._sequence_outputs(observations)
            state = None
        log_probs = None if actions is None else self._log_prob(mean, actions)
        return log_probs, values, state
