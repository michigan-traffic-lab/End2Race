import torch
import torch.nn as nn


class End2RaceMLP(nn.Module):
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    ENCODER_HIDDEN_SIZE = 1260
    ENCODER_OUTPUT_SIZE = 420
    MLP_HIDDEN_SIZE = 128
    SPEED_MASK_PROBABILITY = 0.2
    LIDAR_NORMALIZATION_K = 0.3

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "lidar_normalization_k",
            torch.tensor(self.LIDAR_NORMALIZATION_K, dtype=torch.float32),
        )

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, self.SPEED_EMBEDDING_DIM),
            nn.ReLU(),
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, self.SPEED_EMBEDDING_DIM))

        processed_features = self.NUM_LIDAR_FEATURES + self.SPEED_EMBEDDING_DIM

        # 795,480 parameters, within 0.11% of the 796,320-parameter GRU.
        # Tanh keeps the encoder output bounded like the replaced GRU state.
        self.mlp = nn.Sequential(
            nn.Linear(processed_features, self.ENCODER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.ENCODER_HIDDEN_SIZE, self.ENCODER_OUTPUT_SIZE),
            nn.Tanh(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.ENCODER_OUTPUT_SIZE, self.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.MLP_HIDDEN_SIZE, 2),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_normal_(self.dummy_embedding)

    def encode(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
    ) -> torch.Tensor:
        processed_lidar = 2 * torch.sigmoid(-self.lidar_normalization_k * x)

        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)

        if self.training:
            mask = (
                torch.rand(batch_size, seq_len, 1, device=speed_input.device)
                < self.SPEED_MASK_PROBABILITY
            )
            masked_embedding = self.dummy_embedding.expand(
                batch_size, seq_len, -1
            )
            speed_embedding = torch.where(
                mask, masked_embedding, speed_embedding
            )

        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        return self.mlp(features)

    def forward(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode(x, speed_input)
        return self.output_layer(features)
