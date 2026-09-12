"""Permutation-equivariant Transformer encoder for mass-spectrum peak sets."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .metadata_encoder import CollisionEnergyEncoder


class SpectrumEncoder(nn.Module):
    """Project peak features and contextualize them without index positions."""

    def __init__(
        self,
        input_dim: int,
        d_token_input: int,
        n_heads: int,
        n_layers: int,
        d_token_input_feedforward: int,
        dropout: float,
        use_collision_energy: bool = True,
        collision_energy_max: float = 100.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        self.input_dim = input_dim
        self.use_collision_energy = use_collision_energy
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_token_input),
            nn.GELU(),
            nn.LayerNorm(d_token_input),
        )
        self.collision_energy_encoder = (
            CollisionEnergyEncoder(d_token_input, collision_energy_max)
            if use_collision_energy
            else None
        )
        self.metadata_fusion_norm = (
            nn.LayerNorm(d_token_input) if use_collision_energy else None
        )
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

    def forward(
        self,
        tokens: Tensor,
        padding_mask: Tensor,
        collision_energy: Tensor | None = None,
    ) -> Tensor:
        """Encode a padded peak set, optionally conditioning every real peak on energy."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokens must have shape [batch, peaks, {self.input_dim}]."
            )
        if padding_mask.shape != tokens.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool [batch, peaks].")
        if torch.any(padding_mask.all(dim=1)):
            raise ValueError("Every spectrum must contain at least one real token.")
        projected = self.input_projection(tokens)
        if self.collision_energy_encoder is not None:
            if collision_energy is None:
                collision_energy = tokens.new_full((tokens.shape[0],), torch.nan)
            if collision_energy.shape != (tokens.shape[0],):
                raise ValueError("collision_energy must have shape [batch].")
            assert self.metadata_fusion_norm is not None
            projected = self.metadata_fusion_norm(
                projected
                + self.collision_energy_encoder(collision_energy).unsqueeze(1)
            )
        encoded = self.encoder(
            projected, src_key_padding_mask=padding_mask
        )
        return encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
