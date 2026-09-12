"""Command-line smoke test for predicted and target node representations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from any2graph_v2.data import build_datamodule
from any2graph_v2.models import build_predictor, build_target_encoder
from any2graph_v2.parameter import parse_encoder_experiment_parameters


def main(arguments: Sequence[str] | None = None) -> None:
    configuration = parse_encoder_experiment_parameters(arguments)
    torch.manual_seed(configuration.data.seed)
    data = build_datamodule(configuration.data)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    predictor = build_predictor(configuration.data, configuration.model).eval()
    target_encoder = build_target_encoder(
        d_node_decoder=configuration.model.d_node_decoder,
        parameters=configuration.target_encoder,
    ).eval()
    with torch.inference_mode():
        prediction = predictor.forward_batch(batch)
        target_node_embeddings = target_encoder(batch.graphs)
    print("task/input tokens:", configuration.data.task, tuple(batch.tokens.shape))
    print("predicted graph presence:", tuple(prediction.graph_logits.h.shape))
    print("predicted latent nodes:", tuple(prediction.node_embeddings.shape))
    print("target latent nodes:", tuple(target_node_embeddings.shape))
    print("target padding mask:", tuple((~batch.graphs.h).shape))


if __name__ == "__main__":
    main()
