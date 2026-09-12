"""Lightweight dense bond-aware GNN for target molecular graphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from any2graph_v2.data.dense_data import BatchedDenseData
from any2graph_v2.parameter import (
    VALID_ATOMS_LIST,
    VALID_BOND_TYPES,
    TargetEncoderParameters,
)

from .laplacian_pe import LaplacianPositionalEncoding


class DenseBondMessageLayer(nn.Module):
    """Aggregate relation-specific messages over the four real bond types."""

    def __init__(self, d_node_target: int, n_relations: int, dropout: float) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList(
            nn.Linear(d_node_target, d_node_target, bias=False)
            for _ in range(n_relations)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * d_node_target, d_node_target),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node_target, d_node_target),
        )
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(d_node_target)

    def forward(
        self, node_states: Tensor, bond_weights: Tensor, node_mask: Tensor
    ) -> Tensor:
        relation_states = torch.stack(
            [projection(node_states) for projection in self.relation_linears],
            dim=2,
        )
        messages = torch.einsum("bijk,bjkd->bid", bond_weights, relation_states)
        degree = bond_weights.sum(dim=(2, 3)).clamp_min(1.0).unsqueeze(-1)
        messages = messages / degree
        update = self.update(torch.cat((node_states, messages), dim=-1))
        node_states = self.normalization(node_states + self.dropout(update))
        return node_states.masked_fill(~node_mask.unsqueeze(-1), 0.0)


class TargetGraphEncoder(nn.Module):
    """Map target atoms and bonds to matcher-width latent node embeddings."""

    def __init__(
        self,
        d_node_decoder: int,
        parameters: TargetEncoderParameters | None = None,
        *,
        n_node_classes: int = len(VALID_ATOMS_LIST),
        n_edge_classes: int = 1 + len(VALID_BOND_TYPES),
    ) -> None:
        super().__init__()
        if d_node_decoder < 1:
            raise ValueError("d_node_decoder must be positive.")
        self.configuration = parameters or TargetEncoderParameters()
        if n_node_classes < 1 or n_edge_classes < 2:
            raise ValueError("Graph encoders require node and edge classes.")
        self.n_node_classes = n_node_classes
        self.n_edge_classes = n_edge_classes
        d_node_target = self.configuration.d_node_target
        positional_dim = (
            self.configuration.laplacian_pe_dim
            if self.configuration.use_laplacian_pe
            else 0
        )
        self.laplacian_encoding = (
            LaplacianPositionalEncoding(self.configuration.laplacian_pe_dim)
            if self.configuration.use_laplacian_pe
            else None
        )
        self.atom_projection = nn.Linear(
            n_node_classes + positional_dim, d_node_target
        )
        self.layers = nn.ModuleList(
            DenseBondMessageLayer(
                d_node_target=d_node_target,
                n_relations=n_edge_classes - 1,
                dropout=self.configuration.target_dropout,
            )
            for _ in range(self.configuration.target_layers)
        )
        self.output_projection = nn.Linear(d_node_target, d_node_decoder)
        self.output_normalization = nn.LayerNorm(d_node_decoder)

    def _validate(self, targets: BatchedDenseData) -> None:
        if targets.h.dtype != torch.bool:
            raise TypeError("TargetGraphEncoder requires boolean target h values.")
        expected_nodes = (targets.batchsize, targets.size, self.n_node_classes)
        if "labels" not in targets.nodes or targets.nodes.labels.shape != expected_nodes:
            raise ValueError(f"Target atom labels must have shape {expected_nodes}.")
        expected_edges = (
            targets.batchsize,
            targets.size,
            targets.size,
            self.n_edge_classes,
        )
        if "labels" not in targets.edges or targets.edges.labels.shape != expected_edges:
            raise ValueError(f"Target bond labels must have shape {expected_edges}.")
        if torch.any(targets.h.sum(dim=1) == 0):
            raise ValueError("Every target graph must contain at least one real node.")

    def forward(self, targets: BatchedDenseData) -> Tensor:
        """Return zero-padded matcher-width target-node embeddings.

        Atom and categorical bond fields must follow the fixed project
        vocabularies. Padded target nodes remain exactly zero after every graph
        encoding stage.
        """
        self._validate(targets)
        node_mask = targets.h
        atom_features = targets.nodes.labels
        if self.laplacian_encoding is not None:
            adjacency = targets.edges.labels[..., 1:].sum(dim=-1)
            positional = self.laplacian_encoding(adjacency, node_mask)
            atom_features = torch.cat((atom_features, positional), dim=-1)
        node_states = self.atom_projection(atom_features)
        node_states = node_states.masked_fill(~node_mask.unsqueeze(-1), 0.0)

        real_pairs = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        no_self_loops = ~torch.eye(
            targets.size, dtype=torch.bool, device=targets.h.device
        ).unsqueeze(0)
        edge_mask = (real_pairs & no_self_loops).unsqueeze(-1)
        bond_weights = targets.edges.labels[..., 1:] * edge_mask
        for layer in self.layers:
            node_states = layer(node_states, bond_weights, node_mask)

        output = self.output_normalization(self.output_projection(node_states))
        return output.masked_fill(~node_mask.unsqueeze(-1), 0.0)


def build_target_encoder(
    d_node_decoder: int,
    parameters: TargetEncoderParameters | None = None,
    *,
    n_node_classes: int = len(VALID_ATOMS_LIST),
    n_edge_classes: int = 1 + len(VALID_BOND_TYPES),
) -> nn.Module:
    """Select the target encoder while preserving one matcher interface."""

    configuration = parameters or TargetEncoderParameters()
    if configuration.target_encoder_type == "gnn":
        return TargetGraphEncoder(
            d_node_decoder,
            configuration,
            n_node_classes=n_node_classes,
            n_edge_classes=n_edge_classes,
        )
    raise ValueError(
        f"Unsupported target encoder: {configuration.target_encoder_type!r}."
    )
