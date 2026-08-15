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
    """Recurrent value network over the same observation the actor receives."""

    GRU_HIDDEN_SIZE = 128
    MLP_HIDDEN_SIZE = 128

    def __init__(self, maximum_speed):
        super().__init__()
        self.register_buffer("lidar_normalization_k", torch.tensor(End2Race.LIDAR_NORMALIZATION_K, dtype=torch.float32))
        self.register_buffer("maximum_speed", torch.tensor(maximum_speed, dtype=torch.float32))

        self.gru = nn.GRU(input_size=End2Race.NUM_LIDAR_FEATURES + 1, hidden_size=self.GRU_HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(self.GRU_HIDDEN_SIZE, self.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.MLP_HIDDEN_SIZE, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)

    def initial_hidden(self, batch_size, device):
        return torch.zeros((1, batch_size, self.GRU_HIDDEN_SIZE), dtype=torch.float32, device=device)

    def _features(self, observations):
        lidar = observations[..., :End2Race.NUM_LIDAR_FEATURES]
        speed = observations[..., End2Race.NUM_LIDAR_FEATURES:]
        # Reuse the actor's LiDAR normalization so both networks see the same scale
        return torch.cat((2 * torch.sigmoid(-self.lidar_normalization_k * lidar), speed / self.maximum_speed), dim=-1)

    def step_values(self, observations, hidden):
        """Score one 40 Hz observation per environment and advance the value hidden state."""
        gru_out, hidden = self.gru(self._features(observations)[:, None], hidden)
        return self.output_layer(gru_out)[:, 0, 0], hidden

    def evaluate(self, observations):
        """Replay whole trajectories from a zero hidden state and return per-step values."""
        gru_out = self.gru(self._features(observations), None)[0]
        return self.output_layer(gru_out)[..., 0]
