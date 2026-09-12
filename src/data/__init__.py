"""Task-routed data loading and shared dense graph containers."""

from __future__ import annotations

import lightning.pytorch as pl

from any2graph_v2.parameter import DataParameters

from .coloring import (
    ColoringGraphBatch,
    ColoringGraphDataModule,
    ColoringGraphDataset,
    collate_coloring_graph,
)
from .dense_data import BatchedDenseData, DenseData
from .fingerprint import (
    FingerprintGraphBatch,
    FingerprintGraphDataModule,
    FingerprintGraphDataset,
    FingerprintGraphStore,
    collate_fingerprint_graph,
)
from .massspecgym import (
    MassSpecGymDataModule,
    MassSpecGymDataset,
    SpectrumGraphBatch,
)
from .molecule_graph import MoleculeGraphResult, build_molecular_graph


def build_datamodule(parameters: DataParameters) -> pl.LightningDataModule:
    """Select the input datamodule solely from ``parameters.task``."""

    if parameters.task == "coloring2graph":
        return ColoringGraphDataModule(parameters)
    if parameters.task == "fingerprint2graph":
        return FingerprintGraphDataModule(parameters)
    if parameters.task == "ms2graph":
        return MassSpecGymDataModule(parameters)
    raise ValueError(f"Unsupported task: {parameters.task!r}.")

__all__ = [
    "BatchedDenseData",
    "ColoringGraphBatch",
    "ColoringGraphDataModule",
    "ColoringGraphDataset",
    "DenseData",
    "FingerprintGraphBatch",
    "FingerprintGraphDataModule",
    "FingerprintGraphDataset",
    "FingerprintGraphStore",
    "MassSpecGymDataModule",
    "MassSpecGymDataset",
    "MoleculeGraphResult",
    "SpectrumGraphBatch",
    "build_molecular_graph",
    "build_datamodule",
    "collate_coloring_graph",
    "collate_fingerprint_graph",
]
