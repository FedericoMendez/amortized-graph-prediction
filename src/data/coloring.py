"""Synthetic Coloring image-to-graph dataset and Lightning DataModule."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from any2graph_v2.generate_coloring import DATASET_FORMAT_VERSION
from any2graph_v2.parameter import (
    COLORING_EDGE_CLASSES,
    COLORING_NODE_CLASSES,
    DataParameters,
)

from .dense_data import BatchedDenseData, DenseData


@dataclass
class ColoringGraphBatch:
    """A batch of RGB input images and padded target graphs."""

    images: torch.Tensor
    graphs: BatchedDenseData
    metadata: list[dict[str, Any]]

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "ColoringGraphBatch":
        self.images = self.images.to(device, non_blocking=non_blocking)
        self.graphs.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> "ColoringGraphBatch":
        self.images = self.images.pin_memory()
        self.graphs.pin_memory()
        return self


ColoringItem = tuple[torch.Tensor, DenseData, dict[str, Any]]


class ColoringGraphDataset(Dataset[ColoringItem]):
    """Decode RGB inputs and expose target graphs below a node capacity."""

    def __init__(
        self,
        root: Path,
        split: str,
        n_nodes_max: int,
        *,
        augment: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.n_nodes_max = n_nodes_max
        self.augment = augment
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format_version") != DATASET_FORMAT_VERSION:
            raise ValueError(
                "Unsupported Coloring dataset format version; regenerate the "
                "image-to-graph data with scripts/coloring/generate_array.sh."
            )
        self.image_resolution = int(manifest["image_resolution"])
        self.noise_std = float(manifest["noise_std"])
        self.dataset_seed = int(manifest["seed"])
        self.palette = np.asarray(manifest["rgb_palette"], dtype=np.float32)
        if self.palette.shape != (COLORING_NODE_CLASSES, 3):
            raise ValueError("Coloring RGB palette has an invalid shape.")
        self.node_offsets = np.load(
            root / f"{split}_node_offsets.npy", mmap_mode="r"
        )
        self.edge_offsets = np.load(
            root / f"{split}_edge_offsets.npy", mmap_mode="r"
        )
        self.image_offsets = np.load(
            root / f"{split}_image_offsets.npy", mmap_mode="r"
        )
        if any(
            offsets.ndim != 1
            for offsets in (self.node_offsets, self.edge_offsets, self.image_offsets)
        ):
            raise ValueError("Coloring offsets must be one-dimensional.")
        offset_lengths = {
            len(self.node_offsets),
            len(self.edge_offsets),
            len(self.image_offsets),
        }
        if len(offset_lengths) != 1:
            raise ValueError("Coloring node, edge, and image offset counts disagree.")
        sizes = np.diff(self.node_offsets)
        self.indices = np.flatnonzero((sizes > 0) & (sizes <= n_nodes_max))
        if not len(self.indices):
            raise ValueError(
                f"No {split} graphs fit n_nodes_max={n_nodes_max} in {root}."
            )
        self.colors = np.memmap(
            root / f"{split}_colors.bin",
            mode="r",
            dtype=np.uint8,
            shape=(int(self.node_offsets[-1]),),
        )
        self.edges = np.memmap(
            root / f"{split}_edges.bin",
            mode="r",
            dtype=np.uint16,
            shape=(int(self.edge_offsets[-1]), 2),
        )
        self.regions = np.memmap(
            root / f"{split}_regions.bin",
            mode="r",
            dtype=np.uint8,
            shape=(int(self.image_offsets[-1]),),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """Apply a random flip and quarter-turn rotation to one training image."""

        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
        quarter_turns = int(torch.randint(0, 4, ()).item())
        return torch.rot90(
            image, k=quarter_turns, dims=(-2, -1)
        ).contiguous()

    def region_map(self, source_index: int) -> np.ndarray:
        """Decode one source raster as integer graph-node region IDs."""

        source_count = len(self.image_offsets) - 1
        if source_index < 0 or source_index >= source_count:
            raise IndexError(
                f"Coloring source index {source_index} is outside [0, {source_count})."
            )
        image_start = int(self.image_offsets[source_index])
        image_end = int(self.image_offsets[source_index + 1])
        compressed = np.asarray(self.regions[image_start:image_end]).tobytes()
        region_bytes = zlib.decompress(compressed)
        expected_bytes = self.image_resolution**2
        if len(region_bytes) != expected_bytes:
            raise ValueError(
                f"Corrupt Coloring raster for {self.split}[{source_index}]: "
                f"expected {expected_bytes} bytes, found {len(region_bytes)}."
            )
        return np.frombuffer(region_bytes, dtype=np.uint8).reshape(
            self.image_resolution, self.image_resolution
        )

    def __getitem__(self, index: int) -> ColoringItem:
        source_index = int(self.indices[index])
        node_start = int(self.node_offsets[source_index])
        node_end = int(self.node_offsets[source_index + 1])
        edge_start = int(self.edge_offsets[source_index])
        edge_end = int(self.edge_offsets[source_index + 1])
        colors_array = np.asarray(self.colors[node_start:node_end]).copy()
        colors = torch.from_numpy(colors_array).long()
        edges = torch.from_numpy(
            np.asarray(self.edges[edge_start:edge_end]).copy()
        ).long()
        size = len(colors)
        regions = self.region_map(source_index)
        if int(regions.max()) >= size:
            raise ValueError("Coloring raster references a nonexistent graph node.")
        image = self.palette[colors_array[regions]]
        if self.noise_std:
            split_index = ("train", "val", "test").index(self.split)
            noise_seed = np.random.SeedSequence(
                (self.dataset_seed, split_index, source_index, 1)
            ).generate_state(1, dtype=np.uint64)[0]
            generator = np.random.default_rng(int(noise_seed))
            image = image + generator.normal(
                loc=0.0,
                scale=self.noise_std,
                size=image.shape,
            ).astype(np.float32)
        image_tensor = torch.from_numpy(
            np.clip(image, 0.0, 1.0).transpose(2, 0, 1).copy()
        ).float()
        if self.augment:
            image_tensor = self.transform(image_tensor)
        node_labels = functional.one_hot(
            colors, num_classes=COLORING_NODE_CLASSES
        ).float()
        adjacency = torch.zeros((size, size), dtype=torch.float32)
        if len(edges):
            adjacency[edges[:, 0], edges[:, 1]] = 1.0
            adjacency[edges[:, 1], edges[:, 0]] = 1.0
        edge_indices = adjacency.long()
        edge_labels = functional.one_hot(
            edge_indices, num_classes=COLORING_EDGE_CLASSES
        ).float()
        graph = DenseData(
            size=size,
            nodes={"labels": node_labels},
            edges={"labels": edge_labels, "adjacency": adjacency},
        )
        return image_tensor, graph, {
            "split": self.split,
            "source_index": source_index,
            "graph_size": size,
            "image_resolution": self.image_resolution,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.split,
            "dataset_size": len(self),
            "n_nodes_max": self.n_nodes_max,
            "image_resolution": self.image_resolution,
            "augmentation": self.augment,
        }


def collate_coloring_graph(
    samples: list[ColoringItem], *, n_nodes_max: int
) -> ColoringGraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty list.")
    images, graphs, metadata = zip(*samples)
    return ColoringGraphBatch(
        images=torch.stack(images),
        graphs=BatchedDenseData.from_list(graphs, target_size=n_nodes_max),
        metadata=list(metadata),
    )


class ColoringGraphDataModule(pl.LightningDataModule):
    """Load generated Coloring RGB inputs and graph targets."""

    def __init__(self, parameters: DataParameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.train_dataset: ColoringGraphDataset | None = None
        self.val_dataset: ColoringGraphDataset | None = None
        self.test_dataset: ColoringGraphDataset | None = None

    def prepare_data(self) -> None:
        manifest_path = self.parameters.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing Coloring dataset manifest: {manifest_path}. "
                "Run scripts/coloring/generate_array.sh first."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != DATASET_FORMAT_VERSION:
            raise ValueError(
                "Unsupported Coloring dataset format version. Existing v1 data "
                "is graph-only; regenerate it with scripts/coloring/generate_array.sh."
            )
        if manifest.get("input_modality") != "rgb_image":
            raise ValueError("Coloring input modality must be rgb_image.")
        if manifest.get("node_classes") != COLORING_NODE_CLASSES:
            raise ValueError("Coloring node-class count does not match the model.")
        if manifest.get("edge_classes") != COLORING_EDGE_CLASSES:
            raise ValueError("Coloring edge-class count does not match the model.")
        if self.parameters.n_nodes_max > int(manifest.get("max_nodes", 0)):
            raise ValueError(
                "n_nodes_max exceeds the maximum generated Coloring graph size."
            )

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is None:
            self.prepare_data()
            arguments = (self.parameters.data_dir, self.parameters.n_nodes_max)
            self.train_dataset = ColoringGraphDataset(
                arguments[0], "train", arguments[1], augment=True
            )
            self.val_dataset = ColoringGraphDataset(arguments[0], "val", arguments[1])
            self.test_dataset = ColoringGraphDataset(
                arguments[0], "test", arguments[1]
            )

    def _loader(
        self, dataset: ColoringGraphDataset | None, *, shuffle: bool
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
                collate_coloring_graph,
                n_nodes_max=self.parameters.n_nodes_max,
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
