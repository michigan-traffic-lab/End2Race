import torch
import torch.nn as nn


class End2RaceTransformer(nn.Module):
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    GRU_HIDDEN_SIZE = 420
    MLP_HIDDEN_SIZE = 128
    SPEED_MASK_PROBABILITY = 0.2
    LIDAR_NORMALIZATION_K = 0.3
    # One second of history at the 40 Hz control rate.
    CONTEXT_LENGTH = 40
    # One temporal block of 420-dimensional state, matching the single GRU layer.
    NUM_LAYERS = 1
    NUM_HEADS = 2
    FEEDFORWARD_DIM = 420

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

        # The GRU maps the 210-dimensional observation straight into its 420-dimensional
        # state; a Transformer needs the same width on its input and output, so the
        # observation is projected once instead of being read by three gates.
        self.input_projection = nn.Linear(processed_features, self.GRU_HIDDEN_SIZE)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.GRU_HIDDEN_SIZE,
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
            norm=nn.LayerNorm(self.GRU_HIDDEN_SIZE),
            enable_nested_tensor=False,
        )
        for layer in self.transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)

        # Positions are relative to the start of the context window, so the encoding a
        # token receives is the same during training and at any point of an evaluation
        # run, however long the episode is.
        positions = torch.arange(self.CONTEXT_LENGTH, dtype=torch.float32)[:, None]
        frequencies = 10000.0 ** (
            -torch.arange(0, self.GRU_HIDDEN_SIZE, 2, dtype=torch.float32)
            / self.GRU_HIDDEN_SIZE
        )
        angles = positions * frequencies[None]
        position_encoding = torch.empty(
            self.CONTEXT_LENGTH, self.GRU_HIDDEN_SIZE, dtype=torch.float32
        )
        position_encoding[:, 0::2] = angles.sin()
        position_encoding[:, 1::2] = angles.cos()
        self.register_buffer("position_encoding", position_encoding, persistent=False)

        # The block already emits the 420 dimensions the decoder expects.
        self.feature_projection = nn.Identity()

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

        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        tokens = self.input_projection(features)
        if context is not None:
            tokens = torch.cat([context, tokens], dim=1)
        if tokens.shape[1] > self.CONTEXT_LENGTH:
            tokens = tokens[:, -self.CONTEXT_LENGTH :]

        positioned_tokens = tokens + self.position_encoding[
            : tokens.shape[1]
        ].to(dtype=tokens.dtype)[None]
        # The window never exceeds CONTEXT_LENGTH, so a plain causal mask already
        # limits every step to the last second.
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
