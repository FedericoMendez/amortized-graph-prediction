"""Mass-spectrum acquisition metadata encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CollisionEnergyEncoder(nn.Module):
    """Proposed sinusoidal collision-energy encoding with learned NaN state."""

    def __init__(
        self, dimension: int, max_value: float = 100.0, base: float = 10000.0
    ) -> None:
        super().__init__()
        if dimension <= 0 or dimension % 2 != 0:
            raise ValueError("dimension must be positive and even.")
        if max_value <= 0 or base <= 0:
            raise ValueError("max_value and base must be positive.")
        self.dimension = dimension
        self.max_value = float(max_value)
        self.register_buffer(
            "div_term",
            base
            ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension),
        )
        self.nan_embedding = nn.Parameter(torch.empty(dimension))
        nn.init.normal_(self.nan_embedding, std=0.02)

    def forward(self, collision_energy: Tensor) -> Tensor:
        if collision_energy.ndim != 1:
            raise ValueError("collision_energy must have shape [batch].")
        if not collision_energy.is_floating_point():
            raise TypeError("collision_energy must be floating point.")
        missing = torch.isnan(collision_energy)
        angles = (
            collision_energy.masked_fill(missing, 0.0) / self.max_value
        ).unsqueeze(-1) / self.div_term
        encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(
            -2
        )
        return torch.where(missing.unsqueeze(-1), self.nan_embedding, encoded)
