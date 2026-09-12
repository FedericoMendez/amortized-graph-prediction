"""Transformer encoder for ECFP4 active-bit token sequences."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FingerprintEncoder(nn.Module):
    """Embed Morgan-bit IDs and contextualize them with sequence positions."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_token_input: int,
        n_heads: int,
        n_layers: int,
        d_token_input_feedforward: int,
        dropout: float,
        max_tokens: int,
        use_token_positions: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.use_token_positions = use_token_positions
        self.token_embedding = nn.Embedding(vocab_size, d_token_input)
        position = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_token_input, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_token_input)
        )
        positional = torch.zeros(max_tokens, d_token_input)
        positional[:, 0::2] = torch.sin(position * frequencies)
        positional[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("positional_encoding", positional)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token_input,
            nhead=n_heads,
            dim_feedforward=d_token_input_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_token_input),
            enable_nested_tensor=False,
        )

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        """Encode SOS-prefixed active Morgan-bit IDs as padded token memory."""
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be int64 [batch, substructures].")
        if padding_mask.shape != tokens.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool with the token shape.")
        if tokens.shape[1] > self.max_tokens:
            raise ValueError("Token sequence exceeds configured max_tokens.")
        if torch.any(padding_mask.all(dim=1)):
            raise ValueError("Every fingerprint must contain the SOS token.")
        embedded = self.token_embedding(tokens)
        if self.use_token_positions:
            embedded = embedded + self.positional_encoding[: tokens.shape[1]]
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
