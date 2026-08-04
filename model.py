import torch
import torch.nn as nn

from config import load_project_config


MODEL_CONFIG = load_project_config().model
if not 0.0 <= MODEL_CONFIG.speed_mask_probability <= 1.0:
    raise ValueError("speed_mask_probability must be between 0 and 1")
if min(
    MODEL_CONFIG.lidar_features,
    MODEL_CONFIG.speed_embedding_dim,
    MODEL_CONFIG.gru_hidden_size,
    MODEL_CONFIG.mlp_hidden_size,
) <= 0:
    raise ValueError("model dimensions must be positive")
if MODEL_CONFIG.lidar_normalization_k <= 0:
    raise ValueError("lidar_normalization_k must be positive")


class End2Race(nn.Module):
    NUM_LIDAR_FEATURES = MODEL_CONFIG.lidar_features
    SPEED_EMBEDDING_DIM = MODEL_CONFIG.speed_embedding_dim
    GRU_HIDDEN_SIZE = MODEL_CONFIG.gru_hidden_size
    MLP_HIDDEN_SIZE = MODEL_CONFIG.mlp_hidden_size

    def __init__(self):
        super().__init__()

        self.speed_mask_probability = MODEL_CONFIG.speed_mask_probability
        self.register_buffer(
            "lidar_normalization_k",
            torch.tensor(MODEL_CONFIG.lidar_normalization_k, dtype=torch.float32),
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
                if module.bias is not None:
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
        if x.shape[-1] != self.NUM_LIDAR_FEATURES:
            raise ValueError(
                f"Expected {self.NUM_LIDAR_FEATURES} LiDAR features, got {x.shape[-1]}"
            )
        processed_lidar = 2 * torch.sigmoid(-self.lidar_normalization_k * x)

        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)

        if self.training and self.speed_mask_probability > 0:
            mask = (
                torch.rand(batch_size, seq_len, 1, device=speed_input.device)
                < self.speed_mask_probability
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
