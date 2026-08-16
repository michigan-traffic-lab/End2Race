import torch
import torch.nn as nn
from model import End2Race


class GaussianActor(nn.Module):
    """End2Race with a fixed-variance diagonal Gaussian over steering and desired speed."""

    def __init__(self, checkpoint_path, steering_std, speed_std):
        super().__init__()
        self.model = End2Race()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.register_buffer("action_std", torch.tensor([steering_std, speed_std], dtype=torch.float32))
        self.train()

    def train(self, mode=True):
        # End2Race masks the speed embedding while training, so only the GRU stays in training mode for cuDNN backward
        super().train(False)
        self.model.gru.train(True)
        return self

    def initial_hidden(self, batch_size, device):
        return torch.zeros((1, batch_size, self.model.gru.hidden_size), dtype=torch.float32, device=device)

    def _split(self, observations):
        return observations[..., :End2Race.NUM_LIDAR_FEATURES], observations[..., End2Race.NUM_LIDAR_FEATURES:]

    def _log_prob(self, mean, actions):
        return torch.distributions.Normal(mean, self.action_std).log_prob(actions).sum(-1)

    def act(self, observations, hidden):
        """Sample one 40 Hz action per environment and return its log-probability."""
        lidar, speed = self._split(observations)
        mean, hidden = self.model(lidar[:, None], speed[:, None], hidden)
        mean = mean[:, 0]
        actions = torch.normal(mean, self.action_std.expand_as(mean))
        return actions, self._log_prob(mean, actions), hidden

    def evaluate(self, observations, actions):
        """Replay whole trajectories from a zero hidden state and score the stored actions."""
        lidar, speed = self._split(observations)
        mean = self.model(lidar, speed, None)[0]
        return self._log_prob(mean, actions)


class Critic(nn.Module):
    """BC-pretrained End2Race backbone with a scalar value head."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = End2Race()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.output_layer = nn.Sequential(
            nn.Linear(End2Race.GRU_HIDDEN_SIZE, End2Race.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(End2Race.MLP_HIDDEN_SIZE, 1),
        )
        nn.init.xavier_uniform_(self.model.output_layer[0].weight)
        nn.init.zeros_(self.model.output_layer[0].bias)
        nn.init.zeros_(self.model.output_layer[2].weight)
        nn.init.zeros_(self.model.output_layer[2].bias)
        self.train()

    def train(self, mode=True):
        # Disable speed masking while retaining the GRU training path needed by cuDNN backward
        super().train(False)
        self.model.gru.train(True)
        return self

    def initial_hidden(self, batch_size, device):
        return torch.zeros((1, batch_size, self.model.gru.hidden_size), dtype=torch.float32, device=device)

    def _split(self, observations):
        return observations[..., :End2Race.NUM_LIDAR_FEATURES], observations[..., End2Race.NUM_LIDAR_FEATURES:]

    def step_values(self, observations, hidden):
        """Score one 40 Hz observation per environment and advance the value hidden state."""
        lidar, speed = self._split(observations)
        values, hidden = self.model(lidar[:, None], speed[:, None], hidden)
        return values[:, 0, 0], hidden

    def evaluate(self, observations):
        """Replay whole trajectories from a zero hidden state and return per-step values."""
        lidar, speed = self._split(observations)
        return self.model(lidar, speed, None)[0][..., 0]
