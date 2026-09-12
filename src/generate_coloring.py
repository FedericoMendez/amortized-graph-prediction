"""Generate synthetic Coloring image-to-graph datasets.

Each sample is an L1 Voronoi diagram rendered as a noisy RGB image.  The
target graph has one node per Voronoi region, the region color as its node
class, and an undirected edge whenever two regions share a raster boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt
from tqdm.auto import tqdm

from any2graph_v2.parameter import COLORING_EDGE_CLASSES, COLORING_NODE_CLASSES


DATASET_FORMAT_VERSION = 2
DEFAULT_RESOLUTION_PER_MAX_NODE = 4
DEFAULT_NOISE_STD = 0.05
SPLITS = ("train", "val", "test")
RGB_PALETTE = np.asarray(
    [
        [0.894, 0.102, 0.110],
        [0.216, 0.494, 0.722],
        [0.302, 0.686, 0.290],
        [1.000, 0.851, 0.184],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ColoringSample:
    """Compressed input raster and compact target graph for one sample."""

    colors: np.ndarray
    edges: np.ndarray
    compressed_regions: bytes


def _raster_adjacency(regions: np.ndarray) -> np.ndarray:
    """Return region pairs that share a horizontal or vertical boundary."""

    horizontal = np.stack((regions[:, :-1], regions[:, 1:]), axis=-1).reshape(-1, 2)
    vertical = np.stack((regions[:-1, :], regions[1:, :]), axis=-1).reshape(-1, 2)
    pairs = np.concatenate((horizontal, vertical), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.uint16)
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0).astype(np.uint16, copy=False)


def _greedy_dsatur(adjacency: list[set[int]]) -> list[int]:
    """Produce a fast DSATUR coloring, potentially using more than four colors."""

    colors = [-1] * len(adjacency)
    for _ in colors:
        node = max(
            (index for index, color in enumerate(colors) if color < 0),
            key=lambda index: (
                len({colors[item] for item in adjacency[index] if colors[item] >= 0}),
                len(adjacency[index]),
                -index,
            ),
        )
        unavailable = {colors[item] for item in adjacency[node] if colors[item] >= 0}
        colors[node] = next(
            color for color in range(len(adjacency)) if color not in unavailable
        )
    return colors


def _four_color(edges: np.ndarray, size: int) -> np.ndarray:
    """Find a proper four-coloring with greedy DSATUR and exact fallback."""

    adjacency = [set() for _ in range(size)]
    for left, right in edges.tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)
    greedy = _greedy_dsatur(adjacency)
    if max(greedy, default=0) < COLORING_NODE_CLASSES:
        return np.asarray(greedy, dtype=np.uint8)

    colors = [-1] * size

    def search(colored: int) -> bool:
        if colored == size:
            return True
        node = max(
            (index for index, color in enumerate(colors) if color < 0),
            key=lambda index: (
                len({colors[item] for item in adjacency[index] if colors[item] >= 0}),
                len(adjacency[index]),
                -index,
            ),
        )
        unavailable = {colors[item] for item in adjacency[node] if colors[item] >= 0}
        for color in range(COLORING_NODE_CLASSES):
            if color in unavailable:
                continue
            colors[node] = color
            if search(colored + 1):
                return True
            colors[node] = -1
        return False

    if not search(0):
        raise RuntimeError("Failed to four-color an L1 Voronoi adjacency graph.")
    return np.asarray(colors, dtype=np.uint8)


def _l1_voronoi_regions(
    *, size: int, image_resolution: int, generator: np.random.Generator
) -> np.ndarray:
    """Rasterize an L1 Voronoi partition with one nonempty region per seed."""

    locations = generator.choice(image_resolution**2, size=size, replace=False)
    rows, columns = np.divmod(locations, image_resolution)
    seeds = np.ones((image_resolution, image_resolution), dtype=np.uint8)
    seeds[rows, columns] = 0
    nearest = distance_transform_cdt(
        seeds,
        metric="taxicab",
        return_distances=False,
        return_indices=True,
    )
    seed_ids = np.full((image_resolution, image_resolution), -1, dtype=np.int16)
    seed_ids[rows, columns] = np.arange(size, dtype=np.int16)
    regions = seed_ids[nearest[0], nearest[1]]
    if np.any(regions < 0) or len(np.unique(regions)) != size:
        raise RuntimeError("L1 Voronoi rasterization lost one or more seed regions.")
    return regions.astype(np.uint8, copy=False)


def generate_coloring_sample(
    *, min_nodes: int, max_nodes: int, image_resolution: int, seed: int
) -> ColoringSample:
    """Generate one noisy-image input representation and target graph."""

    generator = np.random.default_rng(seed)
    size = int(generator.integers(min_nodes, max_nodes + 1))
    regions = _l1_voronoi_regions(
        size=size,
        image_resolution=image_resolution,
        generator=generator,
    )
    edges = _raster_adjacency(regions)
    colors = _four_color(edges, size)
    # Numeric color IDs from DSATUR are arbitrary; randomizing their mapping
    # removes a relationship between class ID and coloring search order.
    colors = generator.permutation(COLORING_NODE_CLASSES)[colors].astype(np.uint8)
    return ColoringSample(
        colors=colors,
        edges=edges,
        compressed_regions=zlib.compress(regions.tobytes(), level=6),
    )


def _generate_from_specification(
    specification: tuple[int, int, int, int],
) -> ColoringSample:
    min_nodes, max_nodes, image_resolution, seed = specification
    return generate_coloring_sample(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        image_resolution=image_resolution,
        seed=seed,
    )


def _sample_seed(seed: int, split_index: int, sample_index: int) -> int:
    sequence = np.random.SeedSequence((seed, split_index, sample_index))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _write_split(
    directory: Path,
    split: str,
    count: int,
    *,
    split_index: int,
    min_nodes: int,
    max_nodes: int,
    image_resolution: int,
    seed: int,
    workers: int,
) -> dict[str, float | int]:
    node_offsets = np.zeros(count + 1, dtype=np.int64)
    edge_offsets = np.zeros(count + 1, dtype=np.int64)
    image_offsets = np.zeros(count + 1, dtype=np.int64)
    colors_path = directory / f"{split}_colors.bin"
    edges_path = directory / f"{split}_edges.bin"
    regions_path = directory / f"{split}_regions.bin"
    specifications = (
        (
            min_nodes,
            max_nodes,
            image_resolution,
            _sample_seed(seed, split_index, index),
        )
        for index in range(count)
    )
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        samples = executor.map(
            _generate_from_specification, specifications, chunksize=32
        )
    else:
        samples = map(_generate_from_specification, specifications)

    node_total = 0
    edge_total = 0
    compressed_total = 0
    try:
        with (
            colors_path.open("wb") as color_stream,
            edges_path.open("wb") as edge_stream,
            regions_path.open("wb") as region_stream,
        ):
            for index, sample in enumerate(
                tqdm(samples, total=count, desc=f"generating {split}", unit="samples")
            ):
                sample.colors.tofile(color_stream)
                sample.edges.tofile(edge_stream)
                region_stream.write(sample.compressed_regions)
                node_total += len(sample.colors)
                edge_total += len(sample.edges)
                compressed_total += len(sample.compressed_regions)
                node_offsets[index + 1] = node_total
                edge_offsets[index + 1] = edge_total
                image_offsets[index + 1] = compressed_total
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    np.save(directory / f"{split}_node_offsets.npy", node_offsets)
    np.save(directory / f"{split}_edge_offsets.npy", edge_offsets)
    np.save(directory / f"{split}_image_offsets.npy", image_offsets)
    sizes = np.diff(node_offsets)
    return {
        "samples": count,
        "nodes": int(node_total),
        "edges": int(edge_total),
        "compressed_image_bytes": int(compressed_total),
        "size_mean": float(sizes.mean()) if count else 0.0,
        "size_min": int(sizes.min()) if count else 0,
        "size_max": int(sizes.max()) if count else 0,
    }


def generate_dataset(
    output_dir: Path,
    *,
    min_nodes: int,
    max_nodes: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    seed: int,
    workers: int,
    image_resolution: int | None = None,
    noise_std: float = DEFAULT_NOISE_STD,
) -> dict[str, object]:
    """Generate all splits into a compact, atomically published directory."""

    if min_nodes < 1 or max_nodes < min_nodes:
        raise ValueError("Require 1 <= min_nodes <= max_nodes.")
    if max_nodes > np.iinfo(np.uint8).max:
        raise ValueError("max_nodes exceeds the uint8 region-label format.")
    if any(value < 1 for value in (train_samples, val_samples, test_samples)):
        raise ValueError("Every split must contain at least one sample.")
    if workers < 1:
        raise ValueError("workers must be positive.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if image_resolution is None:
        image_resolution = DEFAULT_RESOLUTION_PER_MAX_NODE * max_nodes
    if image_resolution < 1 or image_resolution**2 < max_nodes:
        raise ValueError("image_resolution must provide at least one pixel per node.")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Remove or rename it explicitly."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        counts = (train_samples, val_samples, test_samples)
        summaries = {
            split: _write_split(
                temporary,
                split,
                count,
                split_index=index,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                image_resolution=image_resolution,
                seed=seed,
                workers=workers,
            )
            for index, (split, count) in enumerate(zip(SPLITS, counts, strict=True))
        }
        manifest: dict[str, object] = {
            "format_version": DATASET_FORMAT_VERSION,
            "generator": "l1_voronoi_rgb_image_to_graph",
            "input_modality": "rgb_image",
            "seed": seed,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "image_resolution": image_resolution,
            "resolution_per_max_node": image_resolution / max_nodes,
            "noise_std": noise_std,
            "rgb_palette": RGB_PALETTE.tolist(),
            "node_classes": COLORING_NODE_CLASSES,
            "edge_classes": COLORING_EDGE_CLASSES,
            "splits": summaries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noisy RGB L1-Voronoi image-to-graph data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=None,
        help=(
            "square image side; default is "
            f"{DEFAULT_RESOLUTION_PER_MAX_NODE} * max_nodes"
        ),
    )
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--train-samples", type=int, default=300_000)
    parser.add_argument("--val-samples", type=int, default=10_000)
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = argument_parser().parse_args(arguments)
    manifest = generate_dataset(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
