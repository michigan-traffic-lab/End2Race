import torch
import torch.nn as nn


class End2RaceTransformer(nn.Module):
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    MODEL_DIM = NUM_LIDAR_FEATURES + SPEED_EMBEDDING_DIM
    FEEDFORWARD_DIM = 420
    OUTPUT_HIDDEN_SIZE = 64
    LIDAR_NORMALIZATION_K = 0.3
    CONTEXT_LENGTH = 20
    NUM_LAYERS = 2
    NUM_HEADS = 3

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.MODEL_DIM,
            nhead=self.NUM_HEADS,
            dim_feedforward=self.FEEDFORWARD_DIM,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.NUM_LAYERS,
            norm=nn.LayerNorm(self.MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.register_buffer(
            "position_encoding",
            self._sinusoidal_encoding(),
            persistent=False,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.MODEL_DIM, self.OUTPUT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.OUTPUT_HIDDEN_SIZE, 2),
        )

        self._initialize_parameters()

    def _sinusoidal_encoding(self) -> torch.Tensor:
        positions = torch.arange(self.CONTEXT_LENGTH, dtype=torch.float32)[:, None]
        frequencies = 10000.0 ** (
            -torch.arange(0, self.MODEL_DIM, 2, dtype=torch.float32)
            / self.MODEL_DIM
        )
        angles = positions * frequencies[None]
        encoding = torch.empty(
            self.CONTEXT_LENGTH, self.MODEL_DIM, dtype=torch.float32
        )
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles.cos()
        return encoding

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)

    def encode(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len, _ = x.shape
        if seq_len > self.CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the {self.CONTEXT_LENGTH}-step "
                "context window; feed sliding windows instead of whole episodes"
            )
        if padding_mask is not None and padding_mask.shape != x.shape[:2]:
            raise ValueError("padding_mask must match the batch and sequence dimensions")
        if context is not None and padding_mask is not None:
            raise ValueError("context and padding_mask are separate inference and training inputs")

        processed_lidar = 2 * torch.sigmoid(-self.lidar_normalization_k * x)
        speed_embedding = self.speed_mlp(speed_input)
        tokens = torch.cat([processed_lidar, speed_embedding], dim=2)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        tokens = tokens[:, -self.CONTEXT_LENGTH :]

        window_length = tokens.shape[1]
        features = self.transformer(
            tokens + self.position_encoding[:window_length].to(dtype=tokens.dtype),
            src_key_padding_mask=padding_mask,
        )

        return features[:, -seq_len:], tokens[:, -(self.CONTEXT_LENGTH - 1) :]

    def forward(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, new_context = self.encode(x, speed_input, context, padding_mask)
        actions = self.output_layer(features)
        return actions, new_context
