"""Task-routed input encoder and shared Any2GraphV2 graph predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from any2graph_v2.data.coloring import ColoringGraphBatch
from any2graph_v2.data.dense_data import BatchedDenseData
from any2graph_v2.data.fingerprint import FingerprintGraphBatch
from any2graph_v2.data.massspecgym import SpectrumGraphBatch
from any2graph_v2.parameter import (
    FINGERPRINT_VOCAB_SIZE,
    COLORING_EDGE_CLASSES,
    COLORING_NODE_CLASSES,
    VALID_ATOMS_LIST,
    VALID_BOND_TYPES,
    DataParameters,
    ModelParameters,
    TargetEncoderParameters,
)

from .coloring_image_encoder import build_coloring_image_encoder
from .graph_heads import MolecularGraphHeads
from .fingerprint_encoder import FingerprintEncoder
from .node_query_decoder import NodeQueryDecoder
from .relationformer_decoder import RelationformerDecoder
from .spectrum_encoder import SpectrumEncoder


@dataclass
class SpectrumGraphPrediction:
    """Latent predicted nodes and graph logits in predicted-slot order."""

    node_embeddings: Tensor
    graph_logits: BatchedDenseData
    relation_embedding: Tensor | None = None


class Any2GraphV2Predictor(nn.Module):
    """Map task-specific inputs to the shared node decoder and graph heads."""

    def __init__(
        self,
        n_nodes: int = 32,
        max_input_tokens: int = 256,
        parameters: ModelParameters | None = None,
        *,
        task: str = "fingerprint2graph",
        spectrum_input_dim: int | None = None,
        target_encoder_parameters: TargetEncoderParameters | None = None,
        feature_diffusion: bool = False,
    ) -> None:
        super().__init__()
        self.configuration = parameters or ModelParameters()
        self.task = task
        self.n_nodes = n_nodes
        if n_nodes < 1:
            raise ValueError("n_nodes must be positive.")
        if task == "coloring2graph":
            self.input_encoder = build_coloring_image_encoder(
                encoder_type=self.configuration.coloring_image_encoder_type,
                d_token_input=self.configuration.d_token_input,
                n_heads=self.configuration.n_heads,
                n_layers=self.configuration.encoder_layers,
                d_token_input_feedforward=(
                    self.configuration.d_token_input_feedforward
                ),
                dropout=self.configuration.dropout,
                max_feature_grid_size=self.configuration.image_feature_grid_size,
            )
        elif task == "fingerprint2graph":
            self.input_encoder: nn.Module = FingerprintEncoder(
                vocab_size=FINGERPRINT_VOCAB_SIZE,
                d_token_input=self.configuration.d_token_input,
                n_heads=self.configuration.n_heads,
                n_layers=self.configuration.encoder_layers,
                d_token_input_feedforward=(
                    self.configuration.d_token_input_feedforward
                ),
                dropout=self.configuration.dropout,
                max_tokens=max_input_tokens,
                use_token_positions=self.configuration.use_token_positions,
            )
        elif task == "ms2graph":
            if spectrum_input_dim is None:
                raise ValueError("spectrum_input_dim is required for task=ms2graph.")
            self.input_encoder = SpectrumEncoder(
                input_dim=spectrum_input_dim,
                d_token_input=self.configuration.d_token_input,
                n_heads=self.configuration.n_heads,
                n_layers=self.configuration.encoder_layers,
                d_token_input_feedforward=(
                    self.configuration.d_token_input_feedforward
                ),
                dropout=self.configuration.dropout,
                use_collision_energy=self.configuration.use_collision_energy,
                collision_energy_max=self.configuration.collision_energy_max,
            )
        else:
            raise ValueError(f"Unsupported task: {task!r}.")
        decoder_class = (
            RelationformerDecoder
            if self.configuration.graph_decoder_type == "relationformer"
            else NodeQueryDecoder
        )
        self.node_decoder = decoder_class(
            n_nodes=n_nodes,
            d_token_input=self.configuration.d_token_input,
            d_node_decoder=self.configuration.d_node_decoder,
            n_heads=self.configuration.n_heads,
            n_layers=self.configuration.decoder_layers,
            d_node_decoder_feedforward=(
                self.configuration.d_node_decoder_feedforward
            ),
            dropout=self.configuration.dropout,
        )
        n_node_classes = (
            COLORING_NODE_CLASSES
            if task == "coloring2graph"
            else len(VALID_ATOMS_LIST)
        )
        n_edge_classes = (
            COLORING_EDGE_CLASSES
            if task == "coloring2graph"
            else 1 + len(VALID_BOND_TYPES)
        )
        self.graph_heads = MolecularGraphHeads(
            d_node_decoder=self.configuration.d_node_decoder,
            d_edge_decoder=self.configuration.d_edge_decoder,
            n_atom_classes=n_node_classes,
            n_bond_classes=n_edge_classes,
            dropout=self.configuration.dropout,
            feature_diffusion=feature_diffusion,
            relationformer=(
                self.configuration.graph_decoder_type == "relationformer"
            ),
        )

    def _decode_memory(
        self, encoded_tokens: Tensor, padding_mask: Tensor
    ) -> SpectrumGraphPrediction:
        relation_embedding = None
        if isinstance(self.node_decoder, RelationformerDecoder):
            decoded = self.node_decoder(encoded_tokens, padding_mask)
            node_embeddings = decoded.node_embeddings
            relation_embedding = decoded.relation_embedding
        else:
            node_embeddings = self.node_decoder(encoded_tokens, padding_mask)
        return SpectrumGraphPrediction(
            node_embeddings=node_embeddings,
            graph_logits=self.graph_heads(
                node_embeddings,
                relation_embedding=relation_embedding,
            ),
            relation_embedding=relation_embedding,
        )

    def forward(
        self,
        tokens: Tensor,
        padding_mask: Tensor,
        collision_energy: Tensor | None = None,
    ) -> SpectrumGraphPrediction:
        """Decode one padded task-specific token batch into dense graph logits.

        ``tokens`` and ``padding_mask`` have batch-first sequence axes. Collision
        energy is required only for ``ms2graph``; passing it to Fingerprint2Graph
        is rejected to prevent silently mixing the two input contracts.
        """
        if self.task == "coloring2graph":
            raise TypeError("Use forward_batch with a Coloring image batch.")
        if self.task == "fingerprint2graph":
            if collision_energy is not None:
                raise ValueError("collision_energy is not used by fingerprint2graph.")
            encoded_tokens = self.input_encoder(tokens, padding_mask)
        else:
            encoded_tokens = self.input_encoder(
                tokens, padding_mask, collision_energy  # type: ignore[call-arg]
            )
        return self._decode_memory(encoded_tokens, padding_mask)

    def forward_batch(
        self, batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch
    ) -> SpectrumGraphPrediction:
        """Predict from a routed batch without reading its target graph fields."""

        if isinstance(batch, ColoringGraphBatch):
            if self.task != "coloring2graph":
                raise TypeError("A Coloring batch requires task=coloring2graph.")
            encoded = self.input_encoder(batch.images)
            padding_mask = torch.zeros(
                encoded.shape[:2], dtype=torch.bool, device=encoded.device
            )
            return self._decode_memory(encoded, padding_mask)
        if isinstance(batch, FingerprintGraphBatch):
            if self.task != "fingerprint2graph":
                raise TypeError("A fingerprint batch requires task=fingerprint2graph.")
            return self(batch.tokens, batch.padding_mask)
        if self.task != "ms2graph":
            raise TypeError("A spectrum batch requires task=ms2graph.")
        return self(batch.tokens, batch.padding_mask, batch.collision_energy)


def build_predictor(
    data_parameters: DataParameters,
    model_parameters: ModelParameters | None = None,
    target_encoder_parameters: TargetEncoderParameters | None = None,
    *,
    feature_diffusion: bool = False,
) -> Any2GraphV2Predictor:
    """Build the task-specific input encoder plus shared prediction pipeline."""

    spectrum_input_dim = None
    if data_parameters.task == "ms2graph":
        spectrum_input_dim = (
            16 if data_parameters.spectrum_representation == "annotated" else 2
        )
    return Any2GraphV2Predictor(
        n_nodes=data_parameters.n_nodes_max,
        max_input_tokens=data_parameters.n_tokens_max,
        parameters=model_parameters,
        task=data_parameters.task,
        spectrum_input_dim=spectrum_input_dim,
        target_encoder_parameters=target_encoder_parameters,
        feature_diffusion=feature_diffusion,
    )
