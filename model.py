import torch
import torch.nn as nn


class End2Race(nn.Module):
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    GRU_HIDDEN_SIZE = 420
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

        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=self.GRU_HIDDEN_SIZE,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.GRU_HIDDEN_SIZE, self.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.MLP_HIDDEN_SIZE, 2),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        nn.init.xavier_normal_(self.dummy_embedding)

    def forward(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        gru_out, last_hidden = self.gru(features, hidden)
        actions = self.output_layer(gru_out)

        return actions, last_hidden
