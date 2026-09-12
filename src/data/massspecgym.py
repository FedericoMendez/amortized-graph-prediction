"""MassSpecGym datasets and Lightning DataModule."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from any2graph_v2.parameter import DataParameters

from .dense_data import BatchedDenseData, DenseData
from .molecule_graph import MoleculeGraphResult, build_molecular_graph
from .spectra import SubformulaTransform, trim_raw_spectrum


@dataclass
class SpectrumGraphBatch:
    """A padded spectrum batch, dense targets, and traceable sample metadata."""

    tokens: Tensor
    padding_mask: Tensor
    collision_energy: Tensor
    graphs: BatchedDenseData
    metadata: list[dict[str, Any]]

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "SpectrumGraphBatch":
        self.tokens = self.tokens.to(device, non_blocking=non_blocking)
        self.padding_mask = self.padding_mask.to(device, non_blocking=non_blocking)
        self.collision_energy = self.collision_energy.to(
            device, non_blocking=non_blocking
        )
        self.graphs.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> "SpectrumGraphBatch":
        self.tokens = self.tokens.pin_memory()
        self.padding_mask = self.padding_mask.pin_memory()
        self.collision_energy = self.collision_energy.pin_memory()
        self.graphs.pin_memory()
        return self


class MassSpecGymStore:
    """Validated shared view of aligned MassSpecGym files."""

    def __init__(self, parameters: DataParameters) -> None:
        self.parameters = parameters
        root = parameters.data_dir
        self.metadata = pd.read_csv(root / "metadata.csv")
        self.unique_smiles = pd.read_csv(root / "unique_smiles.csv")
        split = pd.read_csv(root / "splits" / f"{parameters.split_method}.csv")
        if "unique_smiles_idx" not in self.metadata:
            raise ValueError("metadata.csv is missing unique_smiles_idx.")
        if "smiles" not in self.unique_smiles:
            raise ValueError("unique_smiles.csv is missing smiles.")
        if "fold" not in split:
            raise ValueError("The split file is missing fold.")
        if len(self.metadata) != len(split):
            raise ValueError("Metadata and split files have different lengths.")
        molecule_indices = self.metadata["unique_smiles_idx"].to_numpy(dtype=np.int64)
        if len(molecule_indices) and (
            molecule_indices.min() < 0
            or molecule_indices.max() >= len(self.unique_smiles)
        ):
            raise ValueError("metadata.csv contains an invalid molecule index.")
        self.molecule_indices = molecule_indices
        self.folds = split["fold"].to_numpy()
        self.smiles = self.unique_smiles["smiles"].to_numpy()

        self.annotated: dict[str, list[Any]] | None = None
        self.raw_spectra: np.ndarray | None = None
        if parameters.spectrum_representation == "annotated":
            with (root / "annotated_peaks.json").open() as stream:
                self.annotated = json.load(stream)
            required = ("mz", "intensities", "subformulas")
            if any(key not in self.annotated for key in required):
                raise ValueError("annotated_peaks.json is missing required fields.")
            if any(len(self.annotated[key]) != len(self.metadata) for key in required):
                raise ValueError("Annotated fields are not aligned with metadata.")
        else:
            self.raw_spectra = np.load(root / "spectra.npy", mmap_mode="r")
            if self.raw_spectra.shape[:1] != (len(self.metadata),):
                raise ValueError("spectra.npy is not aligned with metadata.")
            if self.raw_spectra.ndim != 3 or self.raw_spectra.shape[-1] != 2:
                raise ValueError("spectra.npy must have shape [samples, peaks, 2].")

        self.transform = SubformulaTransform()
        self.valid_molecule = np.zeros(len(self.smiles), dtype=bool)
        self.molecule_graphs: list[MoleculeGraphResult | None] = []
        self.policy_counts: Counter[str] = Counter()
        for index, smiles in enumerate(self.smiles):
            result = build_molecular_graph(str(smiles), parameters.graph)
            self.molecule_graphs.append(result)
            if result is None:
                self.policy_counts["removed"] += 1
                continue
            self.valid_molecule[index] = True
            self.policy_counts["valid"] += 1
            if result.scaffold_fallback:
                self.policy_counts["scaffold_fallback"] += 1
            if result.truncated:
                self.policy_counts["truncated"] += 1

    def spectrum_tokens(self, spectrum_index: int) -> Tensor:
        if self.annotated is not None:
            values = self.transform(
                self.annotated["mz"][spectrum_index],
                self.annotated["intensities"][spectrum_index],
                self.annotated["subformulas"][spectrum_index],
            )
        else:
            assert self.raw_spectra is not None
            values = trim_raw_spectrum(np.asarray(self.raw_spectra[spectrum_index]))
        return torch.as_tensor(np.array(values, copy=True), dtype=torch.float32)

    def indices_for_fold(self, fold: str) -> np.ndarray:
        in_fold = self.folds == fold
        eligible = self.valid_molecule[self.molecule_indices]
        return np.flatnonzero(in_fold & eligible)


class MassSpecGymDataset(Dataset[tuple[Tensor, DenseData, dict[str, Any]]]):
    """Spectrum/graph pairs with spectrum- or molecule-level iteration."""

    def __init__(
        self,
        store: MassSpecGymStore,
        fold: str,
        iteration_mode: str,
        seed: int = 0,
    ) -> None:
        if iteration_mode not in {"spectrum", "molecule"}:
            raise ValueError("iteration_mode must be spectrum or molecule.")
        self.store = store
        self.fold = fold
        self.iteration_mode = iteration_mode
        self.rng = np.random.default_rng(seed)
        self.spectrum_indices = store.indices_for_fold(fold)
        grouped: dict[int, list[int]] = defaultdict(list)
        for spectrum_index in self.spectrum_indices:
            molecule_index = int(store.molecule_indices[spectrum_index])
            grouped[molecule_index].append(int(spectrum_index))
        self.spectra_by_molecule = {
            key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()
        }
        self.molecule_indices = np.asarray(sorted(grouped), dtype=np.int64)

    def __len__(self) -> int:
        if self.iteration_mode == "spectrum":
            return len(self.spectrum_indices)
        return len(self.molecule_indices)

    def _spectrum_index(self, index: int) -> int:
        if self.iteration_mode == "spectrum":
            return int(self.spectrum_indices[index])
        molecule_index = int(self.molecule_indices[index])
        return int(self.rng.choice(self.spectra_by_molecule[molecule_index]))

    def reseed(self, seed: int) -> None:
        """Reset molecule-level sampling for a DataLoader worker."""

        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index: int) -> tuple[Tensor, DenseData, dict[str, Any]]:
        spectrum_index = self._spectrum_index(index)
        molecule_index = int(self.store.molecule_indices[spectrum_index])
        result = self.store.molecule_graphs[molecule_index]
        if result is None:
            raise RuntimeError(
                "A molecule accepted during dataset setup failed graph construction."
            )
        metadata = self.store.metadata.iloc[spectrum_index].to_dict()
        metadata.update(
            {
                "fold": self.fold,
                "spectrum_index": spectrum_index,
                "molecule_index": molecule_index,
                "smiles": result.canonical_smiles,
                "target_smiles": result.target_smiles,
                "scaffold_fallback": result.scaffold_fallback,
                "truncated": result.truncated,
                "original_atoms": result.original_atoms,
                "target_atoms_before_truncation": result.target_atoms_before_truncation,
            }
        )
        return self.store.spectrum_tokens(spectrum_index), result.graph, metadata

    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "iteration_mode": self.iteration_mode,
            "dataset_size": len(self),
            "spectra": len(self.spectrum_indices),
            "molecules": len(self.molecule_indices),
            "policy_counts": dict(self.store.policy_counts),
        }


def collate_spectrum_graph(
    samples: Sequence[tuple[Tensor, DenseData, dict[str, Any]]],
    *,
    n_nodes_max: int,
    n_peaks_max: int,
) -> SpectrumGraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty list.")
    spectra, graphs, metadata = zip(*samples)
    feature_dim = spectra[0].shape[1]
    if any(item.ndim != 2 or item.shape[1] != feature_dim for item in spectra):
        raise ValueError("Spectrum features in one batch are inconsistent.")
    peak_count = min(n_peaks_max, max(len(item) for item in spectra))
    tokens = torch.zeros((len(samples), peak_count, feature_dim), dtype=torch.float32)
    padding_mask = torch.ones((len(samples), peak_count), dtype=torch.bool)
    for row, spectrum in enumerate(spectra):
        length = min(len(spectrum), peak_count)
        tokens[row, :length] = spectrum[:length]
        padding_mask[row, :length] = False
    return SpectrumGraphBatch(
        tokens=tokens,
        padding_mask=padding_mask,
        collision_energy=torch.tensor(
            [float(item.get("collision_energy", np.nan)) for item in metadata],
            dtype=torch.float32,
        ),
        graphs=BatchedDenseData.from_list(graphs, target_size=n_nodes_max),
        metadata=list(metadata),
    )


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python from PyTorch's deterministic per-worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker = get_worker_info()
    if worker is not None and isinstance(worker.dataset, MassSpecGymDataset):
        worker.dataset.reseed(worker_seed)


class MassSpecGymDataModule(pl.LightningDataModule):
    """Lightning DataModule for Any2GraphV2 MassSpecGym training."""

    def __init__(self, parameters: DataParameters | None = None) -> None:
        super().__init__()
        self.parameters = parameters or DataParameters()
        self.store: MassSpecGymStore | None = None
        self.train_dataset: MassSpecGymDataset | None = None
        self.val_dataset: MassSpecGymDataset | None = None
        self.test_dataset: MassSpecGymDataset | None = None

    def prepare_data(self) -> None:
        root = self.parameters.data_dir
        required = [
            root / "metadata.csv",
            root / "unique_smiles.csv",
            root / "splits" / f"{self.parameters.split_method}.csv",
        ]
        if self.parameters.spectrum_representation == "annotated":
            required.append(root / "annotated_peaks.json")
        else:
            required.append(root / "spectra.npy")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing MassSpecGym files: {missing}")

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.store is None:
            self.prepare_data()
            self.store = MassSpecGymStore(self.parameters)
        if self.train_dataset is None:
            self.train_dataset = MassSpecGymDataset(
                self.store,
                "train",
                self.parameters.train_iteration_mode,
                seed=self.parameters.seed,
            )
            self.val_dataset = MassSpecGymDataset(
                self.store,
                "val",
                self.parameters.eval_iteration_mode,
                seed=self.parameters.seed + 1,
            )
            self.test_dataset = MassSpecGymDataset(
                self.store,
                "test",
                self.parameters.eval_iteration_mode,
                seed=self.parameters.seed + 2,
            )

    def _loader(self, dataset: MassSpecGymDataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a DataLoader.")
        generator = torch.Generator().manual_seed(self.parameters.seed)
        kwargs: dict[str, Any] = {
            "batch_size": self.parameters.batch_size,
            "shuffle": shuffle,
            "num_workers": self.parameters.num_workers,
            "pin_memory": self.parameters.pin_memory,
            "collate_fn": partial(
                collate_spectrum_graph,
                n_nodes_max=self.parameters.n_nodes_max,
                n_peaks_max=self.parameters.n_peaks_max,
            ),
            "worker_init_fn": seed_worker,
            "generator": generator,
        }
        if self.parameters.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
