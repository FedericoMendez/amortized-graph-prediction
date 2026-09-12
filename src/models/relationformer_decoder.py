"""Any2Graph-style Relationformer decoder with one global relation token."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RelationformerDecoderOutput:
    """Decoded object slots and the shared relation-token state."""

    node_embeddings: Tensor
    relation_embedding: Tensor


class RelationformerDecoderLayer(nn.Module):
    """Pre-normalized DETR decoder layer used by the original Any2Graph code."""

    def __init__(
        self,
        dimension: int,
        n_heads: int,
        feedforward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(dimension, feedforward_dimension)
        self.linear2 = nn.Linear(feedforward_dimension, dimension)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: Tensor,
        memory: Tensor,
        query_positions: Tensor,
        memory_padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.norm1(states)
        queries = normalized + query_positions
        attended = self.self_attention(
            queries, queries, normalized, need_weights=False
        )[0]
        states = states + self.dropout1(attended)

        normalized = self.norm2(states)
        attended = self.cross_attention(
            normalized + query_positions,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        states = states + self.dropout2(attended)

        normalized = self.norm3(states)
        feedforward = self.linear2(
            self.feedforward_dropout(torch.relu(self.linear1(normalized)))
        )
        return states + self.dropout3(feedforward)


class RelationformerDecoder(nn.Module):
    """Decode ``N`` object slots jointly with one Any2Graph virtual node.

    The first of ``N+1`` learned query positions is the global relation token,
    matching the original Any2Graph convention. The remaining states are the
    object/node slots passed to the matcher and reconstruction objective.
    """

    def __init__(
        self,
        n_nodes: int,
        d_token_input: int,
        d_node_decoder: int,
        n_heads: int,
        n_layers: int,
        d_node_decoder_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        self.n_nodes = n_nodes
        self.memory_projection = nn.Linear(d_token_input, d_node_decoder)
        self.query_embeddings = nn.Parameter(
            torch.empty(n_nodes + 1, d_node_decoder)
        )
        self.layers = nn.ModuleList(
            RelationformerDecoderLayer(
                dimension=d_node_decoder,
                n_heads=n_heads,
                feedforward_dimension=d_node_decoder_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_node_decoder)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, memory: Tensor, memory_padding_mask: Tensor
    ) -> RelationformerDecoderOutput:
        if memory.ndim != 3:
            raise ValueError("memory must have shape [batch, tokens, features].")
        if memory_padding_mask.shape != memory.shape[:2]:
            raise ValueError("memory_padding_mask has an incompatible shape.")

        batch_size = memory.shape[0]
        query_positions = self.query_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        states = torch.zeros_like(query_positions)
        projected_memory = self.memory_projection(memory)
        for layer in self.layers:
            states = layer(
                states,
                projected_memory,
                query_positions,
                memory_padding_mask,
            )
        states = self.norm(states)
        return RelationformerDecoderOutput(
            node_embeddings=states[:, 1:],
            relation_embedding=states[:, 0],
        )
