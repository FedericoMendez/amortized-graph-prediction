"""Learned graph-slot queries attending to encoded spectrum peaks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NodeQueryDecoder(nn.Module):
    """Decode exactly one latent vector for each fixed graph slot."""

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
        self.query_embeddings = nn.Parameter(torch.empty(n_nodes, d_node_decoder))
        layer = nn.TransformerDecoderLayer(
            d_model=d_node_decoder,
            nhead=n_heads,
            dim_feedforward=d_node_decoder_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_node_decoder)
        )
        nn.init.normal_(self.query_embeddings, std=0.02)

    def forward(self, memory: Tensor, memory_padding_mask: Tensor) -> Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must have shape [batch, peaks, features].")
        if memory_padding_mask.shape != memory.shape[:2]:
            raise ValueError("memory_padding_mask has an incompatible shape.")
        batch_size = memory.shape[0]
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return self.decoder(
            tgt=queries,
            memory=self.memory_projection(memory),
            memory_key_padding_mask=memory_padding_mask,
        )
