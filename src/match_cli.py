"""Command-line smoke test for the complete matching pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from any2graph_v2.data import build_datamodule
from any2graph_v2.models import build_matcher, build_predictor, build_target_encoder
from any2graph_v2.parameter import parse_matcher_experiment_parameters


def main(arguments: Sequence[str] | None = None) -> None:
    configuration = parse_matcher_experiment_parameters(arguments)
    torch.manual_seed(configuration.data.seed)
    data = build_datamodule(configuration.data)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    predictor = build_predictor(configuration.data, configuration.model).eval()
    target_encoder = build_target_encoder(
        d_node_decoder=configuration.model.d_node_decoder,
        parameters=configuration.target_encoder,
    ).eval()
    matcher = build_matcher(
        d_node_decoder=configuration.model.d_node_decoder,
        n_nodes=configuration.data.n_nodes_max,
        parameters=configuration.matcher,
    ).eval()
    with torch.inference_mode():
        prediction = predictor.forward_batch(batch)
        target_node_embeddings = target_encoder(batch.graphs)
        soft = matcher(
            predicted_node_embeddings=prediction.node_embeddings,
            target_node_embeddings=target_node_embeddings,
            target_padding_mask=~batch.graphs.h,
        )
        hard = matcher.hard_match(
            predicted_node_embeddings=prediction.node_embeddings,
            target_node_embeddings=target_node_embeddings,
            target_padding_mask=~batch.graphs.h,
        )
    print("predicted graph presence:", tuple(prediction.graph_logits.h.shape))
    print("target graph presence:", tuple(batch.graphs.h.shape))
    print("soft matching matrices:", tuple(soft.plan.shape))
    print("hard matching matrices:", tuple(hard.matrices.shape))
    print(
        "maximum row/column marginal errors:",
        float(soft.diagnostics["row_marginal_max_error"]),
        float(soft.diagnostics["column_marginal_max_error"]),
    )
    print("first target-to-predicted permutation:", hard.permutations[0].tolist())


if __name__ == "__main__":
    main()
