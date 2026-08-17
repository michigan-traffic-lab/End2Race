import torch
import torch.nn as nn

from model import End2Race


class ActorCritic(nn.Module):
    """IL-initialized recurrent policy with a shared scalar value head."""

    def __init__(self, checkpoint_path, steering_std, speed_std):
        super().__init__()
        self.model = End2Race()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        self.value_head = nn.Sequential(
            nn.Linear(End2Race.GRU_HIDDEN_SIZE, End2Race.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(End2Race.MLP_HIDDEN_SIZE, 1),
        )
        self.log_action_std = nn.Parameter(
            torch.log(torch.tensor([steering_std, speed_std], dtype=torch.float32))
        )
        self.train()

    def train(self, mode=True):
        # Disable IL speed masking while retaining the GRU training path needed by cuDNN backward
        super().train(False)
        self.model.gru.train(True)
        return self

    @property
    def action_std(self):
        return self.log_action_std.exp()

    def initial_hidden(self, batch_size, device):
        return torch.zeros(
            (1, batch_size, self.model.gru.hidden_size),
            dtype=torch.float32,
            device=device,
        )

    def _split(self, observations):
        return (
            observations[..., :End2Race.NUM_LIDAR_FEATURES],
            observations[..., End2Race.NUM_LIDAR_FEATURES:],
        )

    def _outputs(self, observations, hidden=None):
        lidar, speed = self._split(observations)
        features, hidden = self.model.encode(lidar, speed, hidden)
        return self.model.output_layer(features), self.value_head(features)[..., 0], hidden

    def _log_prob(self, mean, actions):
        return torch.distributions.Normal(mean, self.action_std).log_prob(actions).sum(-1)

    def act(self, observations, hidden):
        """Sample one 40 Hz action per environment and advance the shared hidden state."""
        mean, values, hidden = self._outputs(observations[:, None], hidden)
        mean = mean[:, 0]
        actions = torch.normal(mean, self.action_std.expand_as(mean))
        return actions, mean, self._log_prob(mean, actions), values[:, 0], hidden

    def predict(self, observations, hidden):
        """Return deterministic policy means and advance the shared hidden state."""
        mean, _, hidden = self._outputs(observations[:, None], hidden)
        return mean[:, 0], hidden

    def evaluate(self, observations, actions=None, hidden=None):
        """Replay complete trajectories and return policy and value estimates."""
        mean, values, hidden = self._outputs(observations, hidden)
        log_probs = None if actions is None else self._log_prob(mean, actions)
        return log_probs, values, hidden
