"""Command-line forward/backward smoke test for the objective."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from any2graph_v2.data import build_datamodule
from any2graph_v2.losses import (
    build_graph_reconstruction_objective,
    one_hop_feature_diffusion,
)
from any2graph_v2.models import build_matcher, build_predictor, build_target_encoder
from any2graph_v2.parameter import parse_objective_experiment_parameters
from any2graph_v2.solvers import build_solver


def main(arguments: Sequence[str] | None = None) -> None:
    configuration = parse_objective_experiment_parameters(arguments)
    torch.manual_seed(configuration.data.seed)
    data = build_datamodule(configuration.data)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    predictor = build_predictor(
        configuration.data,
        configuration.model,
        feature_diffusion=configuration.objective.feature_diffusion,
    ).train()
    objective = build_graph_reconstruction_objective(configuration.objective)

    prediction = predictor.forward_batch(batch)
    matched_modules = []
    if configuration.solver.old_solver_approach:
        solver = build_solver(configuration.objective, configuration.solver)
        with torch.no_grad():
            plan = solver(prediction.graph_logits, batch.graphs)
    else:
        target_encoder = build_target_encoder(
            d_node_decoder=configuration.model.d_node_decoder,
            parameters=configuration.target_encoder,
        ).train()
        matcher = build_matcher(
            d_node_decoder=configuration.model.d_node_decoder,
            n_nodes=configuration.data.n_nodes_max,
            parameters=configuration.matcher,
            objective_parameters=configuration.objective,
        ).train()
        target_node_embeddings = target_encoder(batch.graphs)
        matching = matcher(
            predicted_node_embeddings=prediction.node_embeddings,
            target_node_embeddings=target_node_embeddings,
            target_padding_mask=~batch.graphs.h,
            predicted_diffused_features=prediction.graph_logits.nodes.get(
                "diffused_labels"
            ),
            target_diffused_features=(
                one_hop_feature_diffusion(batch.graphs)
                if configuration.objective.feature_diffusion
                else None
            ),
        )
        plan = matching.plan
        matched_modules = [
            ("target_encoder", target_encoder),
            ("matcher", matcher),
        ]
    result = objective(
        predicted_graph=prediction.graph_logits,
        targets=batch.graphs,
        plan=plan,
    )
    result.loss.backward()

    named_parameters = [
        (f"predictor.{name}", parameter)
        for name, parameter in predictor.named_parameters()
    ]
    for prefix, module in matched_modules:
        named_parameters += [
            (f"{prefix}.{name}", parameter)
            for name, parameter in module.named_parameters()
        ]
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in named_parameters
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    maximum_gradient = max(
        float(parameter.grad.abs().max())
        for _, parameter in named_parameters
        if parameter.grad is not None
    )
    print("batch/plan:", tuple(batch.graphs.h.shape), tuple(plan.shape))
    print("total loss:", float(result.loss.detach()))
    print(
        "components:",
        {name: float(value.detach()) for name, value in result.components.items()},
    )
    print(
        "marginal errors:",
        float(result.diagnostics["row_marginal_max_error"].detach()),
        float(result.diagnostics["column_marginal_max_error"].detach()),
    )
    print("parameters with gradients:", len(named_parameters) - len(missing))
    print("maximum absolute gradient:", maximum_gradient)
    print("missing/nonfinite gradients:", missing, nonfinite)
    if missing or nonfinite or not torch.isfinite(result.loss):
        raise RuntimeError("The objective backward smoke test failed.")


if __name__ == "__main__":
    main()
