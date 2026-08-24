import torch
import torch.nn as nn


class End2RaceTransformer(nn.Module):
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

        self.input_projection = nn.Linear(processed_features, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=2,
            dim_feedforward=436,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
            norm=nn.LayerNorm(128),
            enable_nested_tensor=False,
        )
        for layer in self.transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)

        positions = torch.arange(320, dtype=torch.float32)[:, None]
        frequencies = 10000.0 ** (
            -torch.arange(0, 128, 2, dtype=torch.float32) / 128
        )
        angles = positions * frequencies[None]
        position_encoding = torch.empty(320, 128, dtype=torch.float32)
        position_encoding[:, 0::2] = angles.sin()
        position_encoding[:, 1::2] = angles.cos()
        self.register_buffer("position_encoding", position_encoding, persistent=False)

        self.feature_projection = nn.Linear(128, self.GRU_HIDDEN_SIZE)

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

        nn.init.xavier_normal_(self.dummy_embedding)

    def encode(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: torch.Tensor | None = None,
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
        tokens = self.input_projection(features)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)

        positioned_tokens = tokens + self.position_encoding[
            : tokens.shape[1]
        ].to(dtype=tokens.dtype)[None]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        features = self.transformer(
            positioned_tokens,
            mask=causal_mask,
            is_causal=True,
        )
        features = self.feature_projection(features)
        return features[:, -seq_len:], tokens[:, -319:]

    def forward(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, new_context = self.encode(x, speed_input, context)
        actions = self.output_layer(features)

        return actions, new_context
