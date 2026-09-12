"""Lightweight molecular graph heads over latent node slots."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from any2graph_v2.data.dense_data import BatchedDenseData


class MolecularGraphHeads(nn.Module):
    """Predict presence, atoms, and one symmetric categorical bond tensor."""

    diagonal_logit = 20.0

    def __init__(
        self,
        d_node_decoder: int,
        d_edge_decoder: int,
        n_atom_classes: int,
        n_bond_classes: int,
        dropout: float,
        feature_diffusion: bool = False,
        relationformer: bool = False,
    ) -> None:
        super().__init__()
        if n_atom_classes < 1 or n_bond_classes < 2:
            raise ValueError("Graph heads require atom and bond classes.")

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_node_decoder, d_node_decoder),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_node_decoder, output_dim),
            )

        self.relationformer = relationformer
        self.presence_head = head(1)
        self.atom_head = head(n_atom_classes)
        self.feature_diffusion_head = (
            head(n_atom_classes) if feature_diffusion else None
        )
        if relationformer:
            self.edge_projection = nn.Sequential(
                nn.Linear(2 * d_node_decoder, 2 * d_node_decoder),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_node_decoder, 2 * d_node_decoder),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_node_decoder, d_edge_decoder),
            )
            edge_hidden = max(d_edge_decoder // 2, 1)
            self.bond_head = nn.Sequential(
                nn.Linear(d_edge_decoder, edge_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden, edge_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden, n_bond_classes),
            )
        else:
            self.edge_projection = nn.Sequential(
                nn.Linear(d_node_decoder, d_edge_decoder),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.bond_head = nn.Sequential(
                nn.Linear(d_edge_decoder, d_edge_decoder),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_edge_decoder, n_bond_classes),
            )

    def forward(
        self,
        node_embeddings: Tensor,
        *,
        relation_embedding: Tensor | None = None,
    ) -> BatchedDenseData:
        """Produce dense logits with symmetric bonds and forced no-bond diagonals.

        Returned ``h`` contains presence logits; node labels contain atom logits;
        edge labels contain the five-class bond logits; and adjacency is derived
        from the categorical bond distribution rather than predicted separately.
        """
        if node_embeddings.ndim != 3:
            raise ValueError(
                "node_embeddings must have shape [batch, nodes, d_node_decoder]."
            )
        if self.relationformer:
            expected = (node_embeddings.shape[0], node_embeddings.shape[2])
            if relation_embedding is None or relation_embedding.shape != expected:
                raise ValueError(
                    "Relationformer graph heads require relation_embedding "
                    "with shape [batch, d_node_decoder]."
                )
        elif relation_embedding is not None:
            raise ValueError(
                "relation_embedding is only accepted by Relationformer graph heads."
            )

        presence_logits = self.presence_head(node_embeddings).squeeze(-1)
        atom_logits = self.atom_head(node_embeddings)
        if relation_embedding is not None:
            relation_nodes = relation_embedding.unsqueeze(1).expand(
                -1, node_embeddings.shape[1], -1
            )
            edge_input = torch.cat((node_embeddings, relation_nodes), dim=-1)
        else:
            edge_input = node_embeddings
        edge_nodes = self.edge_projection(edge_input)
        pair_features = edge_nodes.unsqueeze(2) + edge_nodes.unsqueeze(1)
        bond_logits = self.bond_head(pair_features)

        n_nodes = node_embeddings.shape[1]
        upper_triangle = torch.triu(
            torch.ones(
                (n_nodes, n_nodes), dtype=torch.bool, device=node_embeddings.device
            )
        ).view(1, n_nodes, n_nodes, 1)
        averaged_logits = (bond_logits + bond_logits.transpose(1, 2)) * 0.5
        bond_logits = torch.where(
            upper_triangle, averaged_logits, averaged_logits.transpose(1, 2)
        )
        diagonal = torch.eye(
            n_nodes, dtype=torch.bool, device=node_embeddings.device
        ).view(1, n_nodes, n_nodes, 1)
        diagonal_values = torch.full_like(bond_logits, -self.diagonal_logit)
        diagonal_values[..., 0] = self.diagonal_logit
        bond_logits = torch.where(diagonal, diagonal_values, bond_logits)

        adjacency_logits = torch.logsumexp(bond_logits[..., 1:], dim=-1)
        adjacency_logits = adjacency_logits - bond_logits[..., 0]
        nodes = {"labels": atom_logits}
        if self.feature_diffusion_head is not None:
            nodes["diffused_labels"] = self.feature_diffusion_head(node_embeddings)
        return BatchedDenseData(
            h=presence_logits,
            nodes=nodes,
            edges={"labels": bond_logits, "adjacency": adjacency_logits},
        )
