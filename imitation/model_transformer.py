import torch
import torch.nn as nn


class End2RaceTransformer(nn.Module):
    NUM_LIDAR_FEATURES = 180
    SPEED_EMBEDDING_DIM = 30
    GRU_HIDDEN_SIZE = 420
    MLP_HIDDEN_SIZE = 128
    SPEED_MASK_PROBABILITY = 0.2
    LIDAR_NORMALIZATION_K = 0.3
    # One second of attention history at the 40 Hz control rate.
    CONTEXT_LENGTH = 40
    # Episodes are 8 seconds long, so positions never exceed this while training.
    MAX_POSITIONS = 320
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

        positions = torch.arange(self.MAX_POSITIONS, dtype=torch.float32)[:, None]
        frequencies = 10000.0 ** (
            -torch.arange(0, self.GRU_HIDDEN_SIZE, 2, dtype=torch.float32)
            / self.GRU_HIDDEN_SIZE
        )
        angles = positions * frequencies[None]
        position_encoding = torch.empty(
            self.MAX_POSITIONS, self.GRU_HIDDEN_SIZE, dtype=torch.float32
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

    def _attention_mask(self, seq_len, device, dtype):
        # Causal, and additionally limited to the last CONTEXT_LENGTH steps, so a whole
        # episode can be trained in one pass while each step still decides from one
        # second of history. With a single layer this is exactly what the policy
        # computes online from its sliding window.
        positions = torch.arange(seq_len, device=device)
        allowed = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] > positions[:, None] - self.CONTEXT_LENGTH
        )
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, float("-inf"))

    def encode(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: tuple[torch.Tensor, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, int]]:
        batch_size, seq_len, _ = x.shape

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

        cached_tokens, window_start = (None, 0) if context is None else context
        if cached_tokens is not None:
            tokens = torch.cat([cached_tokens, tokens], dim=1)
        window_length = tokens.shape[1]
        if window_length > self.MAX_POSITIONS:
            raise ValueError(
                f"Window of {window_length} steps exceeds the {self.MAX_POSITIONS} "
                "positions covered by the position encoding"
            )

        # Positions stay absolute so that a step evaluated online receives the encoding
        # it was trained with. Runs longer than an episode hold at the final window,
        # which is a position pattern every training episode ends on.
        encoding_start = max(0, min(window_start, self.MAX_POSITIONS - window_length))
        positioned_tokens = tokens + self.position_encoding[
            encoding_start : encoding_start + window_length
        ].to(dtype=tokens.dtype)[None]
        features = self.transformer(
            positioned_tokens,
            mask=self._attention_mask(window_length, tokens.device, tokens.dtype),
        )
        features = self.feature_projection(features)

        retained = min(window_length, self.CONTEXT_LENGTH - 1)
        new_context = (
            tokens[:, window_length - retained :],
            window_start + window_length - retained,
        )
        return features[:, -seq_len:], new_context

    def forward(
        self,
        x: torch.Tensor,
        speed_input: torch.Tensor,
        context: tuple[torch.Tensor, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, int]]:
        features, new_context = self.encode(x, speed_input, context)
        actions = self.output_layer(features)

        return actions, new_context
