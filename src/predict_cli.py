"""Command-line smoke prediction through the pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from any2graph_v2.data import build_datamodule
from any2graph_v2.models import build_predictor
from any2graph_v2.parameter import parse_experiment_parameters


def main(arguments: Sequence[str] | None = None) -> None:
    configuration = parse_experiment_parameters(arguments)
    torch.manual_seed(configuration.data.seed)
    data = build_datamodule(configuration.data)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    model = build_predictor(configuration.data, configuration.model).eval()
    with torch.inference_mode():
        prediction = model.forward_batch(batch)
    graph = prediction.graph_logits
    print("task/input tokens:", configuration.data.task, tuple(batch.tokens.shape))
    print("latent nodes:", tuple(prediction.node_embeddings.shape))
    print("presence logits:", tuple(graph.h.shape))
    print("atom logits:", tuple(graph.nodes.labels.shape))
    print("bond logits:", tuple(graph.edges.labels.shape))
    print("adjacency logits:", tuple(graph.edges.adjacency.shape))


if __name__ == "__main__":
    main()
