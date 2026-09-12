"""Transport-plan marginal regularization ported from corrected GRALE."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_plan(plan: Tensor) -> None:
    if plan.ndim != 3:
        raise ValueError("plan must have shape [batch, predicted, target].")
    if not plan.is_floating_point():
        raise TypeError("plan must be floating point.")
    if not torch.isfinite(plan).all():
        raise ValueError("plan must contain only finite values.")
    if torch.any(plan < 0):
        raise ValueError("plan must be non-negative.")


class MarginalKL(nn.Module):
    """GRALE ``sum(m - 1 - log(m))`` over rows and columns."""

    @staticmethod
    def _loss(marginal: Tensor) -> Tensor:
        return (-torch.log(marginal) + marginal - 1.0).sum(dim=-1)

    def forward(self, plan: Tensor) -> Tensor:
        _validate_plan(plan)
        return self._loss(plan.sum(dim=1)) + self._loss(plan.sum(dim=2))
