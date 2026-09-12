"""Hard-aligned molecular graph reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from any2graph_v2.data.dense_data import BatchedDenseData


@dataclass
class GraphMetricResult:
    """Batch means and per-graph values, kept as tensors for Lightning logging."""

    metrics: dict[str, Tensor]
    per_graph: dict[str, Tensor]


class HardGraphMetrics(nn.Module):
    """Compare graph logits and targets after a nondifferentiable hard alignment.

    Presence is evaluated on every padded slot. Atom labels are evaluated only
    on real target nodes, and bonds only on unordered pairs of real target
    nodes. Consequently padding logits cannot improve atom or bond accuracy.
    """

    def __init__(self, presence_threshold: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= presence_threshold <= 1.0:
            raise ValueError("presence_threshold must be in [0, 1].")
        self.presence_threshold = presence_threshold

    @staticmethod
    def _validate(predicted: BatchedDenseData, target: BatchedDenseData) -> None:
        if predicted.h.shape != target.h.shape:
            raise ValueError("Predicted and target graph sizes must agree.")
        if target.h.dtype != torch.bool or not predicted.h.is_floating_point():
            raise TypeError("Expected floating presence logits and boolean targets.")
        if "labels" not in predicted.nodes or "labels" not in target.nodes:
            raise ValueError("Both graphs require node label tensors.")
        if "labels" not in predicted.edges or "labels" not in target.edges:
            raise ValueError("Both graphs require edge label tensors.")
        if predicted.nodes.labels.shape != target.nodes.labels.shape:
            raise ValueError("Predicted and target node label shapes must agree.")
        if predicted.edges.labels.shape != target.edges.labels.shape:
            raise ValueError("Predicted and target edge label shapes must agree.")

    def forward(
        self, predicted: BatchedDenseData, target: BatchedDenseData
    ) -> GraphMetricResult:
        self._validate(predicted, target)
        target_mask = target.h
        predicted_mask = predicted.h.sigmoid() >= self.presence_threshold

        presence_errors = (predicted_mask != target_mask).sum(dim=1).float()
        atom_mismatch = (
            predicted.nodes.labels.argmax(dim=-1)
            != target.nodes.labels.argmax(dim=-1)
        )
        atom_errors = (atom_mismatch & target_mask).sum(dim=1).float()

        predicted_bonds = predicted.edges.labels.argmax(dim=-1)
        target_bonds = target.edges.labels.argmax(dim=-1)
        pair_mask = target_mask.unsqueeze(2) & target_mask.unsqueeze(1)
        upper_triangle = torch.triu(
            torch.ones(
                target.size,
                target.size,
                dtype=torch.bool,
                device=target.h.device,
            ),
            diagonal=1,
        )
        evaluated_pairs = pair_mask & upper_triangle
        bond_errors = (
            (predicted_bonds != target_bonds) & evaluated_pairs
        ).sum(dim=(1, 2)).float()

        target_node_count = target_mask.sum(dim=1).float()
        predicted_node_count = predicted_mask.sum(dim=1).float()
        evaluated_bond_pairs = evaluated_pairs.sum(dim=(1, 2)).float()
        edit_like = presence_errors + atom_errors + bond_errors
        exact = (edit_like == 0).float()
        atom_accuracy = 1.0 - atom_errors / target_node_count.clamp_min(1.0)
        bond_accuracy = 1.0 - bond_errors / evaluated_bond_pairs.clamp_min(1.0)

        per_graph = {
            "presence_errors": presence_errors,
            "atom_errors": atom_errors,
            "bond_errors": bond_errors,
            "edit_like": edit_like,
            "exact_reconstruction": exact,
            "node_count_mae": (predicted_node_count - target_node_count).abs(),
            "atom_accuracy": atom_accuracy,
            "bond_accuracy": bond_accuracy,
        }
        return GraphMetricResult(
            metrics={name: values.mean() for name, values in per_graph.items()},
            per_graph=per_graph,
        )


def proper_coloring_rate(
    predicted: BatchedDenseData, presence_threshold: float = 0.5
) -> Tensor:
    """Return the fraction of predicted graphs with no monochromatic edge."""

    predicted_mask = predicted.h.sigmoid() >= presence_threshold
    node_colors = predicted.nodes.labels.argmax(dim=-1)
    predicted_edges = predicted.edges.labels.argmax(dim=-1) != 0
    real_pairs = predicted_mask.unsqueeze(2) & predicted_mask.unsqueeze(1)
    upper_triangle = torch.triu(
        torch.ones(
            predicted.size,
            predicted.size,
            dtype=torch.bool,
            device=predicted.h.device,
        ),
        diagonal=1,
    )
    same_color = node_colors.unsqueeze(2) == node_colors.unsqueeze(1)
    conflicts = predicted_edges & real_pairs & upper_triangle & same_color
    return (~conflicts.any(dim=(1, 2))).float().mean()
