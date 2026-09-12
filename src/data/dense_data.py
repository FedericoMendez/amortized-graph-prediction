"""Dense graph containers adapted from GRALE's read-only reference code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


class EasyDict(dict[str, Tensor]):
    """Dictionary whose tensor values are also available as attributes."""

    def __getattr__(self, name: str) -> Tensor:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Tensor) -> None:
        self[name] = value


def _pad_dimension(tensor: Tensor, dimension: int, amount: int) -> Tensor:
    if amount == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dimension] = amount
    zeros = tensor.new_zeros(shape)
    return torch.cat((tensor, zeros), dim=dimension)


class DenseData:
    """One dense graph with a real-node indicator and named feature tensors."""

    def __init__(
        self,
        size: int,
        h: Tensor | None = None,
        nodes: Mapping[str, Tensor] | None = None,
        edges: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.size = size
        self.h = h if h is not None else torch.ones(size, dtype=torch.bool)
        self.nodes = EasyDict(nodes or {})
        self.edges = EasyDict(edges or {})
        self._validate()

    def _validate(self) -> None:
        if self.h.shape != (self.size,):
            raise ValueError(f"h must have shape ({self.size},), got {self.h.shape}.")
        for name, tensor in self.nodes.items():
            if tensor.shape[0] != self.size:
                raise ValueError(f"Node tensor {name!r} has incompatible shape.")
        for name, tensor in self.edges.items():
            if tensor.shape[:2] != (self.size, self.size):
                raise ValueError(f"Edge tensor {name!r} has incompatible shape.")

    def pad_(self, new_size: int) -> "DenseData":
        if new_size < self.size:
            raise ValueError("new_size cannot be smaller than the graph.")
        amount = new_size - self.size
        self.h = _pad_dimension(self.h, 0, amount)
        for name, tensor in self.nodes.items():
            self.nodes[name] = _pad_dimension(tensor, 0, amount)
        for name, tensor in self.edges.items():
            tensor = _pad_dimension(tensor, 0, amount)
            self.edges[name] = _pad_dimension(tensor, 1, amount)
        self.size = new_size
        return self

    def clone(self) -> "DenseData":
        return DenseData(
            self.size,
            self.h.clone(),
            {name: value.clone() for name, value in self.nodes.items()},
            {name: value.clone() for name, value in self.edges.items()},
        )


class BatchedDenseData:
    """Batch of equally padded dense graphs using GRALE's tensor convention."""

    def __init__(
        self,
        h: Tensor,
        nodes: Mapping[str, Tensor] | None = None,
        edges: Mapping[str, Tensor] | None = None,
    ) -> None:
        if h.ndim != 2:
            raise ValueError("Batched h must have shape [batch, nodes].")
        self.batchsize, self.size = h.shape
        self.h = h
        self.nodes = EasyDict(nodes or {})
        self.edges = EasyDict(edges or {})
        self._validate()

    def _validate(self) -> None:
        for name, tensor in self.nodes.items():
            if tensor.shape[:2] != (self.batchsize, self.size):
                raise ValueError(f"Node tensor {name!r} has incompatible shape.")
        for name, tensor in self.edges.items():
            if tensor.shape[:3] != (self.batchsize, self.size, self.size):
                raise ValueError(f"Edge tensor {name!r} has incompatible shape.")

    @classmethod
    def from_list(
        cls, graphs: Sequence[DenseData], target_size: int | None = None
    ) -> "BatchedDenseData":
        if not graphs:
            raise ValueError("Cannot batch an empty graph list.")
        size = target_size if target_size is not None else max(g.size for g in graphs)
        if size < max(g.size for g in graphs):
            raise ValueError("target_size is smaller than a graph in the batch.")
        node_keys = tuple(graphs[0].nodes)
        edge_keys = tuple(graphs[0].edges)
        if any(tuple(g.nodes) != node_keys or tuple(g.edges) != edge_keys for g in graphs):
            raise ValueError("All graphs must have identical node and edge fields.")
        padded = [graph.clone().pad_(size) for graph in graphs]
        return cls(
            h=torch.stack([graph.h for graph in padded]),
            nodes={
                name: torch.stack([graph.nodes[name] for graph in padded])
                for name in node_keys
            },
            edges={
                name: torch.stack([graph.edges[name] for graph in padded])
                for name in edge_keys
            },
        )

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "BatchedDenseData":
        self.h = self.h.to(device, non_blocking=non_blocking)
        for name, value in self.nodes.items():
            self.nodes[name] = value.to(device, non_blocking=non_blocking)
        for name, value in self.edges.items():
            self.edges[name] = value.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> "BatchedDenseData":
        self.h = self.h.pin_memory()
        for name, value in self.nodes.items():
            self.nodes[name] = value.pin_memory()
        for name, value in self.edges.items():
            self.edges[name] = value.pin_memory()
        return self

    def clone(self) -> "BatchedDenseData":
        return BatchedDenseData(
            self.h.clone(),
            {name: value.clone() for name, value in self.nodes.items()},
            {name: value.clone() for name, value in self.edges.items()},
        )

    def permute_(self, permutation_matrices: Tensor) -> "BatchedDenseData":
        """Soft-align predicted order ``p`` to target order ``t``.

        ``permutation_matrices[b, p, t]`` follows the project-wide convention.
        """

        expected = (self.batchsize, self.size, self.size)
        if permutation_matrices.shape != expected:
            raise ValueError(f"Expected permutation shape {expected}.")
        if not self.h.is_floating_point():
            raise TypeError("Soft permutation requires floating-point graph values.")
        self.h = torch.einsum("bpt,bp->bt", permutation_matrices, self.h)
        for name, value in self.nodes.items():
            self.nodes[name] = torch.einsum(
                "bpt,bp...->bt...", permutation_matrices, value
            )
        for name, value in self.edges.items():
            self.edges[name] = torch.einsum(
                "bpt,bqu,bpq...->btu...",
                permutation_matrices,
                permutation_matrices,
                value,
            )
        return self

    def align_(self, permutations: Sequence[Sequence[int]]) -> "BatchedDenseData":
        """Hard-align so target position ``t`` takes predicted index ``perm[t]``."""

        indices = torch.as_tensor(permutations, device=self.h.device, dtype=torch.long)
        if indices.shape != (self.batchsize, self.size):
            raise ValueError("Hard permutations have incompatible shape.")
        self.h = torch.gather(self.h, 1, indices)
        for name, value in self.nodes.items():
            index = indices.reshape(self.batchsize, self.size, *([1] * (value.ndim - 2)))
            index = index.expand(self.batchsize, self.size, *value.shape[2:])
            self.nodes[name] = torch.gather(value, 1, index)
        for name, value in self.edges.items():
            index_rows = indices.reshape(
                self.batchsize, self.size, 1, *([1] * (value.ndim - 3))
            ).expand(self.batchsize, self.size, self.size, *value.shape[3:])
            aligned = torch.gather(value, 1, index_rows)
            index_cols = indices.reshape(
                self.batchsize, 1, self.size, *([1] * (value.ndim - 3))
            ).expand(self.batchsize, self.size, self.size, *value.shape[3:])
            self.edges[name] = torch.gather(aligned, 2, index_cols)
        return self

    def __len__(self) -> int:
        return self.batchsize

    def __getitem__(self, index: int) -> DenseData:
        return DenseData(
            self.size,
            self.h[index],
            {name: value[index] for name, value in self.nodes.items()},
            {name: value[index] for name, value in self.edges.items()},
        )
