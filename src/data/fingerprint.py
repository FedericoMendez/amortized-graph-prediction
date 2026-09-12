"""Fingerprint2Graph dataset and Lightning DataModule."""

from __future__ import annotations

import csv
from array import array
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from any2graph_v2.parameter import (
    FINGERPRINT_SOS_TOKEN_ID,
    MORGAN_FINGERPRINT_RADIUS,
    MORGAN_FINGERPRINT_SIZE,
    DataParameters,
    GraphParameters,
)
from any2graph_v2.preprocess_fingerprints import (
    FINGERPRINT_GRAPH_METADATA_COLUMNS,
    has_fingerprint_graph_metadata,
    upgrade_fingerprint_graph_csv,
)

from .dense_data import BatchedDenseData, DenseData
from .molecule_graph import build_molecular_graph


@dataclass
class FingerprintGraphBatch:
    """Padded substructure-token sequences and dense molecular targets."""

    tokens: Tensor
    padding_mask: Tensor
    graphs: BatchedDenseData
    metadata: list[dict[str, Any]]

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "FingerprintGraphBatch":
        self.tokens = self.tokens.to(device, non_blocking=non_blocking)
        self.padding_mask = self.padding_mask.to(device, non_blocking=non_blocking)
        self.graphs.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> "FingerprintGraphBatch":
        self.tokens = self.tokens.pin_memory()
        self.padding_mask = self.padding_mask.pin_memory()
        self.graphs.pin_memory()
        return self


class FingerprintGraphStore:
    """Byte-offset index over a preprocessed CSV, optionally strict-filtered.

    Strict loading uses the persistent graph metadata emitted by preprocessing.
    Older CSVs are upgraded once, atomically, before their index is built. This
    prevents subsequent training jobs from reparsing every SMILES with RDKit.
    """

    def __init__(
        self,
        path: Path,
        graph_parameters: GraphParameters | None = None,
        *,
        metadata_workers: int = 1,
    ) -> None:
        self.path = path
        self.filtered_invalid: Counter[str] = Counter()
        self.metadata_upgrade_counts: Counter[str] = Counter()
        self.metadata_available = False
        self._ensure_strict_metadata(graph_parameters, metadata_workers)
        offsets = {fold: array("Q") for fold in ("train", "val", "test")}
        print(f"[data] building Fingerprint2Graph byte-offset index: {path}", flush=True)
        with path.open("rb") as stream:
            header_line = stream.readline().decode("utf-8")
            self.columns = next(csv.reader([header_line]))
            if not {"smiles", "fold"} <= set(self.columns):
                raise ValueError(f"{path} must contain 'smiles' and 'fold' columns.")
            fold_column = self.columns.index("fold")
            metadata_columns = {
                name: self.columns.index(name)
                for name in FINGERPRINT_GRAPH_METADATA_COLUMNS
                if name in self.columns
            }
            self.metadata_available = has_fingerprint_graph_metadata(self.columns)
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                values = next(csv.reader([line.decode("utf-8")]))
                if len(values) != len(self.columns):
                    raise ValueError(f"Malformed CSV row at byte {offset} in {path}.")
                fold = values[fold_column]
                if fold not in offsets:
                    raise ValueError(f"Unknown fold {fold!r} at byte {offset}.")
                if (
                    graph_parameters is not None
                    and graph_parameters.remove_invalid_molecules
                    and not self._strict_row_is_eligible(
                        values, metadata_columns, graph_parameters
                    )
                ):
                    self.filtered_invalid[fold] += 1
                    continue
                offsets[fold].append(offset)
        self.offsets = offsets
        print(
            "[data] Fingerprint2Graph index ready: "
            + ", ".join(f"{fold}={len(value)}" for fold, value in offsets.items())
            + (
                f"; strict_filtered={sum(self.filtered_invalid.values())}"
                if self.filtered_invalid
                else ""
            ),
            flush=True,
        )

    def _ensure_strict_metadata(
        self, graph_parameters: GraphParameters | None, workers: int
    ) -> None:
        """Upgrade old CSVs only when strict filtering needs reusable facts."""

        if graph_parameters is None or not graph_parameters.remove_invalid_molecules:
            return
        with self.path.open(newline="") as stream:
            columns = next(csv.reader([stream.readline()]))
        if not has_fingerprint_graph_metadata(columns):
            self.metadata_upgrade_counts = upgrade_fingerprint_graph_csv(
                self.path, workers=workers
            )

    @staticmethod
    def _strict_row_is_eligible(
        values: list[str],
        metadata_columns: dict[str, int],
        parameters: GraphParameters,
    ) -> bool:
        """Apply strict full-molecule or scaffold policy without RDKit parsing."""

        if len(metadata_columns) != len(FINGERPRINT_GRAPH_METADATA_COLUMNS):
            raise RuntimeError("Strict Fingerprint2Graph loading requires metadata.")
        try:
            graph_valid = values[metadata_columns["graph_valid"]].strip().lower()
            graph_valid = graph_valid in {"1", "true", "yes"}
            size_name = (
                "molecule_scaffold_size" if parameters.scaffold else "molecule_size"
            )
            target_size = int(values[metadata_columns[size_name]])
        except (IndexError, ValueError) as error:
            raise ValueError("Invalid Fingerprint2Graph graph metadata row.") from error
        return graph_valid and 0 < target_size <= parameters.n_nodes_max

    def row(self, fold: str, index: int, stream: Any) -> dict[str, str]:
        """Read one indexed CSV row from a caller-owned binary stream."""
        stream.seek(self.offsets[fold][index])
        values = next(csv.reader([stream.readline().decode("utf-8")]))
        return dict(zip(self.columns, values, strict=True))


class FingerprintGraphDataset(Dataset[tuple[Tensor, DenseData, dict[str, Any]]]):
    """Generate ECFP4 tokens and a graph from each preprocessed SMILES row."""

    def __init__(
        self, store: FingerprintGraphStore, fold: str, parameters: DataParameters
    ) -> None:
        self.store = store
        self.fold = fold
        self.parameters = parameters
        self._stream: Any = None
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_FINGERPRINT_RADIUS,
            fpSize=MORGAN_FINGERPRINT_SIZE,
        )

    def __len__(self) -> int:
        return len(self.store.offsets[self.fold])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_stream"] = None
        return state

    def __getitem__(self, index: int) -> tuple[Tensor, DenseData, dict[str, Any]]:
        """Generate Morgan tokens and the matching dense target graph for one row."""
        if self._stream is None:
            self._stream = self.store.path.open("rb")
        row = self.store.row(self.fold, index, self._stream)
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid preprocessed SMILES in {self.fold} row {index}.")
        graph_result = build_molecular_graph(smiles, self.parameters.graph)
        if graph_result is None:
            raise ValueError(f"Unsupported molecule in {self.fold} row {index}.")
        active_bits = list(self.generator.GetFingerprint(mol).GetOnBits())
        tokens = torch.tensor(
            [
                FINGERPRINT_SOS_TOKEN_ID,
                *active_bits[: self.parameters.n_tokens_max - 1],
            ],
            dtype=torch.long,
        )
        metadata: dict[str, Any] = dict(row)
        metadata.update(
            {
                "fold_index": index,
                "smiles": graph_result.canonical_smiles,
                "fingerprint_bits": len(active_bits),
                "fingerprint_truncated": len(active_bits) + 1
                > self.parameters.n_tokens_max,
            }
        )
        return tokens, graph_result.graph, metadata

    def summary(self) -> dict[str, Any]:
        """Return the fold name and current number of indexed examples."""
        return {"fold": self.fold, "dataset_size": len(self)}


def collate_fingerprint_graph(
    samples: list[tuple[Tensor, DenseData, dict[str, Any]]], *, n_nodes_max: int
) -> FingerprintGraphBatch:
    """Pad variable-length fingerprint tokens and dense targets into one batch."""
    if not samples:
        raise ValueError("Cannot collate an empty list.")
    sequences, graphs, metadata = zip(*samples)
    length = max(map(len, sequences))
    tokens = torch.zeros((len(samples), length), dtype=torch.long)
    padding_mask = torch.ones((len(samples), length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = sequence
        padding_mask[row, : len(sequence)] = False
    return FingerprintGraphBatch(
        tokens=tokens,
        padding_mask=padding_mask,
        graphs=BatchedDenseData.from_list(graphs, target_size=n_nodes_max),
        metadata=list(metadata),
    )


class FingerprintGraphDataModule(pl.LightningDataModule):
    """Load one preprocessed CSV containing train/val/test fold labels."""

    def __init__(self, parameters: DataParameters | None = None) -> None:
        super().__init__()
        self.parameters = parameters or DataParameters()
        self.store: FingerprintGraphStore | None = None
        self.train_dataset: FingerprintGraphDataset | None = None
        self.val_dataset: FingerprintGraphDataset | None = None
        self.test_dataset: FingerprintGraphDataset | None = None

    def prepare_data(self) -> None:
        """Check that the required preprocessed CSV exists without loading it."""
        if not self.parameters.data_file.is_file():
            raise FileNotFoundError(
                f"Missing preprocessed Fingerprint2Graph file: {self.parameters.data_file}. "
                "Run any2graph-preprocess-fingerprints first."
            )

    def setup(self, stage: str | None = None) -> None:
        """Build the reusable store and train/validation/test dataset views once."""
        del stage
        if self.train_dataset is None:
            self.prepare_data()
            self.store = FingerprintGraphStore(
                self.parameters.data_file,
                self.parameters.graph,
                metadata_workers=max(1, self.parameters.num_workers),
            )
            self.train_dataset = FingerprintGraphDataset(
                self.store, "train", self.parameters
            )
            self.val_dataset = FingerprintGraphDataset(
                self.store, "val", self.parameters
            )
            self.test_dataset = FingerprintGraphDataset(
                self.store, "test", self.parameters
            )

    def _loader(
        self, dataset: FingerprintGraphDataset | None, *, shuffle: bool
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a DataLoader.")
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.parameters.batch_size,
            "shuffle": shuffle,
            "num_workers": self.parameters.num_workers,
            "pin_memory": self.parameters.pin_memory,
            "collate_fn": partial(
                collate_fingerprint_graph, n_nodes_max=self.parameters.n_nodes_max
            ),
            "generator": torch.Generator().manual_seed(self.parameters.seed),
        }
        if self.parameters.num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
