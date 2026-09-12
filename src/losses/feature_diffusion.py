"""Any2Graph-style one-hop node-feature diffusion utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from any2graph_v2.data.dense_data import BatchedDenseData


def one_hop_feature_diffusion(graphs: BatchedDenseData) -> Tensor:
    """Return ``A @ F`` for real target nodes, with padded entries kept zero.

    ``F`` is the categorical node-label matrix and ``A`` is the stored binary
    adjacency. This matches the original Any2Graph ``+FD`` target while making
    its implicit zero-padding assumptions explicit.
    """

    if "labels" not in graphs.nodes or "adjacency" not in graphs.edges:
        raise ValueError("Feature diffusion requires node labels and adjacency.")
    features = graphs.nodes.labels
    adjacency = graphs.edges.adjacency.to(features.dtype)
    if features.ndim != 3 or adjacency.shape != (
        graphs.batchsize,
        graphs.size,
        graphs.size,
    ):
        raise ValueError("Feature diffusion received incompatible graph tensors.")

    real = graphs.h.to(features.dtype)
    features = features * real.unsqueeze(-1)
    adjacency = adjacency * real.unsqueeze(1) * real.unsqueeze(2)
    return torch.bmm(adjacency, features) * real.unsqueeze(-1)


def pairwise_squared_l2(predicted: Tensor, target: Tensor) -> Tensor:
    """Return all squared Euclidean costs with axes ``[batch,predicted,target]``."""

    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError("Feature tensors must have shape [batch, nodes, features].")
    if predicted.shape[0] != target.shape[0]:
        raise ValueError("Predicted and target feature batch sizes must agree.")
    if predicted.shape[2] != target.shape[2]:
        raise ValueError("Predicted and target feature widths must agree.")
    predicted_norm = predicted.square().sum(dim=-1)
    target_norm = target.square().sum(dim=-1)
    cross = torch.bmm(predicted, target.transpose(1, 2))
    squared_distance = (
        predicted_norm.unsqueeze(2) + target_norm.unsqueeze(1) - 2 * cross
    )
    return squared_distance.clamp_min(0)
