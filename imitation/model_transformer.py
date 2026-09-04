import torch
import torch.nn as nn


class End2RaceTransformer(nn.Module):
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    ENCODER_OUTPUT_SIZE = NUM_LIDAR_FEATURES + SPEED_EMBEDDING_DIM
    FEEDFORWARD_DIM = 420
    MLP_HIDDEN_SIZE = 128
    SPEED_MASK_PROBABILITY = 0.2
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
        self.dummy_embedding = nn.Parameter(torch.randn(1, self.SPEED_EMBEDDING_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.ENCODER_OUTPUT_SIZE,
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
            norm=nn.LayerNorm(self.ENCODER_OUTPUT_SIZE),
            enable_nested_tensor=False,
        )
        self.register_buffer(
            "position_encoding",
            self._sinusoidal_encoding(),
            persistent=False,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.ENCODER_OUTPUT_SIZE, self.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.MLP_HIDDEN_SIZE, 2),
        )

        self._initialize_parameters()

    def _sinusoidal_encoding(self) -> torch.Tensor:
        # Positions are relative to the start of the context window, so a token carries
        # the same encoding during training and at any point of an evaluation run.
        positions = torch.arange(self.CONTEXT_LENGTH, dtype=torch.float32)[:, None]
        frequencies = 10000.0 ** (
            -torch.arange(0, self.ENCODER_OUTPUT_SIZE, 2, dtype=torch.float32)
            / self.ENCODER_OUTPUT_SIZE
        )
        angles = positions * frequencies[None]
        encoding = torch.empty(
            self.CONTEXT_LENGTH, self.ENCODER_OUTPUT_SIZE, dtype=torch.float32
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

        nn.init.xavier_normal_(self.dummy_embedding)

    def encode(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the {self.CONTEXT_LENGTH}-step "
                "context window; feed sliding windows instead of whole episodes"
            )

        processed_lidar = 2 * torch.sigmoid(-self.lidar_normalization_k * x)

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

        tokens = torch.cat([processed_lidar, speed_embedding], dim=2)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        tokens = tokens[:, -self.CONTEXT_LENGTH :]

        window_length = tokens.shape[1]
        position_encoding = self.position_encoding[:window_length]
        # The window never exceeds CONTEXT_LENGTH, so a plain causal mask already
        # limits every step to half a second of history.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            window_length,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        features = self.transformer(
            tokens + position_encoding.to(dtype=tokens.dtype),
            mask=causal_mask,
            is_causal=True,
        )

        return features[:, -seq_len:], tokens[:, -(self.CONTEXT_LENGTH - 1) :]

    def forward(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, new_context = self.encode(x, speed_input, context)
        actions = self.output_layer(features)

        return actions, new_context
