"""Command-line smoke test for the task-routed data layer."""

from __future__ import annotations

from collections.abc import Sequence

from any2graph_v2.data import build_datamodule
from any2graph_v2.parameter import parse_data_parameters


def main(arguments: Sequence[str] | None = None) -> None:
    parameters = parse_data_parameters(arguments)
    data = build_datamodule(parameters)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    assert data.train_dataset is not None
    print("train:", data.train_dataset.summary())
    print("task/input tokens:", parameters.task, tuple(batch.tokens.shape))
    print("token padding mask:", tuple(batch.padding_mask.shape))
    print("graph presence:", tuple(batch.graphs.h.shape))
    print("atom targets:", tuple(batch.graphs.nodes.labels.shape))
    print("bond targets:", tuple(batch.graphs.edges.labels.shape))


if __name__ == "__main__":
    main()
