"""Export deterministic Coloring input/prediction/ground-truth examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from any2graph_v2.checkpointing import load_checkpoint
from any2graph_v2.data.coloring import (
    ColoringGraphDataModule,
    ColoringGraphDataset,
    collate_coloring_graph,
)
from any2graph_v2.data.dense_data import DenseData
from any2graph_v2.metrics import HardGraphMetrics
from any2graph_v2.train_cli import build_system


@dataclass(frozen=True)
class ColoringExampleParameters:
    """Runtime-only settings for qualitative checkpoint evaluation."""

    checkpoint: Path
    output_dir: Path = Path("paper_figures/coloring_examples")
    data_dir: Path | None = None
    split: str = "test"
    indices: tuple[int, ...] = (0, 1, 2, 3)
    selection: str = "indices"
    num_examples: int = 4
    device: str = "auto"
    presence_threshold: float | None = None
    formats: tuple[str, ...] = ("png", "pdf")
    output_name: str = "coloring_examples"
    dpi: int = 300


@dataclass(frozen=True)
class DisplayGraph:
    """Hard node/edge decisions used by the qualitative renderer."""

    presence: np.ndarray
    colors: np.ndarray
    edges: np.ndarray


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed Coloring examples as input image, hard-aligned prediction, "
            "and ground-truth graph panels"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_figures/coloring_examples")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="override the Coloring data directory stored in the checkpoint",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="fixed zero-based indices within the selected filtered split",
    )
    parser.add_argument(
        "--selection",
        choices=("indices", "distinct_targets"),
        default="indices",
        help="use exact indices or scan deterministically for distinct target graphs",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="number of examples when selection=distinct_targets",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--presence-threshold",
        type=float,
        help="override the checkpoint's sigmoid node-presence threshold",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"]
    )
    parser.add_argument("--output-name", default="coloring_examples")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def parse_parameters(arguments: list[str] | None = None) -> ColoringExampleParameters:
    parser = argument_parser()
    values = vars(parser.parse_args(arguments))
    indices = tuple(values["indices"])
    formats = tuple(dict.fromkeys(values["formats"]))
    if not indices or any(index < 0 for index in indices):
        parser.error("indices must contain one or more non-negative values")
    if len(set(indices)) != len(indices):
        parser.error("indices must not contain duplicates")
    if values["presence_threshold"] is not None and not (
        0.0 <= values["presence_threshold"] <= 1.0
    ):
        parser.error("presence-threshold must be in [0, 1]")
    if values["dpi"] < 1:
        parser.error("dpi must be positive")
    if values["num_examples"] < 1:
        parser.error("num-examples must be positive")
    if (
        not values["output_name"]
        or Path(values["output_name"]).name != values["output_name"]
    ):
        parser.error("output-name must be a plain filename without directories")
    values["indices"] = indices
    values["formats"] = formats
    return ColoringExampleParameters(**values)


def _coloring_target_identity(sample: tuple[Any, ...]) -> tuple[bytes, bytes]:
    """Identify an exact labeled target graph independently of its source image."""

    graph = sample[1]
    node_labels = graph.nodes.labels.argmax(dim=-1).detach().cpu().numpy()
    edge_labels = graph.edges.labels.argmax(dim=-1).detach().cpu().numpy()
    return node_labels.tobytes(), edge_labels.tobytes()


def _select_coloring_samples(
    dataset: ColoringGraphDataset, parameters: ColoringExampleParameters
) -> tuple[list[int], list[Any]]:
    if parameters.selection == "indices":
        invalid = [index for index in parameters.indices if index >= len(dataset)]
        if invalid:
            raise IndexError(
                f"{parameters.split} indices {invalid} exceed dataset size {len(dataset)}."
            )
        return list(parameters.indices), [dataset[index] for index in parameters.indices]

    selected_indices: list[int] = []
    samples: list[Any] = []
    seen: set[tuple[bytes, bytes]] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        identity = _coloring_target_identity(sample)
        if identity in seen:
            continue
        seen.add(identity)
        selected_indices.append(index)
        samples.append(sample)
        if len(samples) == parameters.num_examples:
            return selected_indices, samples
    raise ValueError(
        f"Only {len(samples)} distinct target graphs were found in "
        f"the {parameters.split} split; requested {parameters.num_examples}."
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _decode_graph(graph: DenseData, *, presence_threshold: float | None) -> DisplayGraph:
    if graph.h.dtype == torch.bool:
        presence = graph.h
    else:
        if presence_threshold is None:
            raise ValueError("Predicted graphs require a presence threshold.")
        presence = graph.h.sigmoid() >= presence_threshold
    colors = graph.nodes.labels.argmax(dim=-1)
    edges = graph.edges.labels.argmax(dim=-1) != 0
    edge_mask = presence.unsqueeze(1) & presence.unsqueeze(0)
    edges = (edges | edges.T) & edge_mask
    edges.fill_diagonal_(False)
    return DisplayGraph(
        presence=presence.detach().cpu().numpy().astype(bool, copy=False),
        colors=colors.detach().cpu().numpy(),
        edges=edges.detach().cpu().numpy().astype(bool, copy=False),
    )


def _region_centroids(regions: np.ndarray, graph_size: int) -> np.ndarray:
    """Map target node IDs to normalized image-region centroids."""

    if regions.ndim != 2 or graph_size < 1:
        raise ValueError("Expected a 2D region map and a positive graph size.")
    height, width = regions.shape
    positions = np.zeros((graph_size, 2), dtype=np.float64)
    for node in range(graph_size):
        rows, columns = np.nonzero(regions == node)
        if not len(rows):
            raise ValueError(f"Coloring region map has no pixels for node {node}.")
        positions[node] = (
            columns.mean() / max(width - 1, 1),
            1.0 - rows.mean() / max(height - 1, 1),
        )
    return positions


def _display_positions(regions: np.ndarray, graph_size: int, capacity: int) -> np.ndarray:
    """Use region centroids for target slots and fixed rows for surplus slots."""

    if capacity < graph_size:
        raise ValueError("Graph capacity cannot be smaller than the target graph.")
    positions = np.zeros((capacity, 2), dtype=np.float64)
    positions[:graph_size] = _region_centroids(regions, graph_size)
    extra_count = capacity - graph_size
    for offset in range(extra_count):
        row, column = divmod(offset, 10)
        row_count = min(10, extra_count - row * 10)
        positions[graph_size + offset] = (
            (column + 1) / (row_count + 1),
            -0.10 - 0.12 * row,
        )
    return positions


def _adaptive_marker_area(
    node_count: int, *, base_area: float, minimum_area: float
) -> float:
    """Shrink marker area for dense graphs while preserving small-graph labels."""

    reference_nodes = 8
    density_scale = np.sqrt(reference_nodes / max(node_count, reference_nodes))
    return max(minimum_area, base_area * float(density_scale))


def _draw_graph(
    axis: Any,
    graph: DisplayGraph,
    positions: np.ndarray,
    *,
    target_size: int,
    title: str,
    mark_missing_targets: bool,
    palette: np.ndarray,
) -> None:
    visible_nodes = np.flatnonzero(graph.presence)
    if mark_missing_targets:
        visible_nodes = np.union1d(visible_nodes, np.arange(target_size))
    if not len(visible_nodes):
        visible_nodes = np.arange(target_size)
    visible_positions = positions[visible_nodes]
    present_count = int(graph.presence.sum())
    marker_area = _adaptive_marker_area(
        present_count,
        base_area=150.0,
        minimum_area=55.0,
    )
    marker_scale = np.sqrt(marker_area / 150.0)
    label_size = max(4.2, 6.5 * float(marker_scale))
    x_limits = (
        min(-0.06, float(visible_positions[:, 0].min()) - 0.12),
        max(1.06, float(visible_positions[:, 0].max()) + 0.12),
    )
    y_limits = (
        min(-0.06, float(visible_positions[:, 1].min()) - 0.12),
        max(1.06, float(visible_positions[:, 1].max()) + 0.12),
    )
    axis.set_title(title, fontsize=10, pad=7)
    axis.set_aspect("equal")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_facecolor("#F7F7F7")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    for left, right in np.argwhere(np.triu(graph.edges, k=1)):
        axis.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="#555555",
            linewidth=1.35,
            alpha=0.85,
            zorder=1,
        )

    if mark_missing_targets:
        missing = np.flatnonzero(~graph.presence[:target_size])
        if len(missing):
            axis.scatter(
                positions[missing, 0],
                positions[missing, 1],
                s=max(38.0, 0.6 * marker_area),
                marker="x",
                color="#B2182B",
                linewidth=1.8,
                zorder=2,
            )

    present = np.flatnonzero(graph.presence)
    if not len(present):
        return
    extra = present >= target_size
    edge_colors = np.where(extra, "#B2182B", "#222222")
    axis.scatter(
        positions[present, 0],
        positions[present, 1],
        s=marker_area,
        c=palette[graph.colors[present]],
        edgecolors=edge_colors,
        linewidths=np.where(extra, 2.2, 1.0),
        zorder=3,
    )
    for node in present:
        color = palette[graph.colors[node]]
        text_color = "black" if float(color.mean()) > 0.65 else "white"
        axis.text(
            positions[node, 0],
            positions[node, 1],
            str(node),
            ha="center",
            va="center",
            fontsize=label_size,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )


def _render_figure(
    records: list[dict[str, Any]],
    *,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError(
            "Qualitative export requires matplotlib. Run 'uv sync --frozen'."
        ) from error

    figure, axes = plt.subplots(
        len(records),
        3,
        figsize=(10.8, 3.15 * len(records)),
        squeeze=False,
    )
    palette = records[0]["palette"]
    for row, record in enumerate(records):
        input_axis, prediction_axis, target_axis = axes[row]
        input_axis.imshow(record["image"])
        input_axis.set_title("Input", fontsize=10, pad=7)
        input_axis.set_xticks([])
        input_axis.set_yticks([])
        for spine in input_axis.spines.values():
            spine.set_visible(False)

        _draw_graph(
            prediction_axis,
            record["prediction"],
            record["positions"],
            target_size=record["target_size"],
            title="Prediction",
            mark_missing_targets=True,
            palette=palette,
        )
        _draw_graph(
            target_axis,
            record["target"],
            record["positions"],
            target_size=record["target_size"],
            title="Ground truth",
            mark_missing_targets=False,
            palette=palette,
        )

    legend_handles = [
        Patch(facecolor=palette[index], edgecolor="#222222", label=f"Color {index}")
        for index in range(len(palette))
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color="#B2182B",
                label="Missing target node",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#B2182B",
                markeredgewidth=2,
                label="Extra predicted node",
            ),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0), h_pad=1.2, w_pad=1.0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for format_name in formats:
        path = output_stem.with_suffix(f".{format_name}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def _render_separate_figures(
    records: list[dict[str, Any]],
    *,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    """Export every RGB input and graph as an independent paper-ready file."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Qualitative export requires matplotlib. Run 'uv sync --frozen'."
        ) from error

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ordinal, record in enumerate(records):
        file_prefix = (
            f"{output_stem.name}-example-{ordinal:02d}-"
            f"index-{record['dataset_index']}"
        )
        for component in ("input", "prediction", "ground-truth"):
            figure, axis = plt.subplots(figsize=(4.8, 4.8))
            if component == "input":
                axis.imshow(record["image"])
                axis.set_title("Input", fontsize=10, pad=7)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            elif component == "prediction":
                _draw_graph(
                    axis,
                    record["prediction"],
                    record["positions"],
                    target_size=record["target_size"],
                    title="Prediction",
                    mark_missing_targets=True,
                    palette=record["palette"],
                )
            else:
                _draw_graph(
                    axis,
                    record["target"],
                    record["positions"],
                    target_size=record["target_size"],
                    title="Ground truth",
                    mark_missing_targets=False,
                    palette=record["palette"],
                )
            figure.tight_layout(pad=0.8)
            for format_name in formats:
                path = output_stem.parent / f"{file_prefix}-{component}.{format_name}"
                figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
                paths.append(path)
            plt.close(figure)
    return paths


def export_coloring_examples(parameters: ColoringExampleParameters) -> dict[str, Any]:
    """Run fixed-example inference and write figures plus reproducibility metadata."""

    checkpoint, stored = load_checkpoint(parameters.checkpoint)
    if stored.data.task != "coloring2graph":
        raise ValueError("Qualitative Coloring export requires task=coloring2graph.")
    device = _resolve_device(parameters.device)
    data_dir = parameters.data_dir or stored.data.data_dir
    example_count = (
        len(parameters.indices)
        if parameters.selection == "indices"
        else parameters.num_examples
    )
    data_parameters = replace(
        stored.data,
        data_dir=data_dir,
        data_file=data_dir / "manifest.json",
        batch_size=example_count,
        num_workers=0,
        pin_memory=False,
    )
    data = ColoringGraphDataModule(data_parameters)
    data.prepare_data()
    dataset = ColoringGraphDataset(
        data_parameters.data_dir,
        parameters.split,
        data_parameters.n_nodes_max,
    )
    selected_indices, samples = _select_coloring_samples(dataset, parameters)
    batch = collate_coloring_graph(samples, n_nodes_max=data_parameters.n_nodes_max)
    configuration = replace(stored, data=data_parameters)
    system = build_system(configuration)
    system.load_state_dict(checkpoint["state_dict"], strict=True)
    system.to(device).eval()
    batch.to(device)
    if (
        system.old_solver_approach
        and system.solver_parameters.solver_type == "mirror"
        and device.type != "cuda"
    ):
        raise RuntimeError("Mirror-solver checkpoints require --device cuda.")

    threshold = (
        parameters.presence_threshold
        if parameters.presence_threshold is not None
        else stored.training.presence_threshold
    )

    with torch.inference_mode():
        prediction, target_embeddings, plan, _ = system._soft_objective(batch)
        aligned = system.hard_aligned_prediction(
            prediction, target_embeddings, plan, batch.graphs
        )
        metric_result = HardGraphMetrics(threshold).to(device)(aligned, batch.graphs)
    records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for batch_index, ((image, _, metadata), dataset_index) in enumerate(
        zip(samples, selected_indices, strict=True)
    ):
        source_index = int(metadata["source_index"])
        target_size = int(metadata["graph_size"])
        regions = dataset.region_map(source_index)
        predicted_graph = _decode_graph(
            aligned[batch_index], presence_threshold=threshold
        )
        target_graph = _decode_graph(batch.graphs[batch_index], presence_threshold=None)
        positions = _display_positions(
            regions, target_size, data_parameters.n_nodes_max
        )
        per_graph_metrics = {
            name: float(values[batch_index].detach().cpu())
            for name, values in metric_result.per_graph.items()
        }
        predicted_size = int(predicted_graph.presence.sum())
        record = {
            "image": image.permute(1, 2, 0).numpy(),
            "prediction": predicted_graph,
            "target": target_graph,
            "positions": positions,
            "palette": dataset.palette,
            "split": parameters.split,
            "dataset_index": dataset_index,
            "source_index": source_index,
            "target_size": target_size,
            "predicted_size": predicted_size,
            "metrics": per_graph_metrics,
        }
        records.append(record)
        manifest_records.append(
            {
                "dataset_index": dataset_index,
                "source_index": source_index,
                "target_identity": "sha256:"
                + hashlib.sha256(
                    b"".join(_coloring_target_identity(samples[batch_index]))
                ).hexdigest(),
                "target_size": target_size,
                "predicted_size": predicted_size,
                "metrics": per_graph_metrics,
            }
        )

    output_stem = parameters.output_dir / parameters.output_name
    figure_paths = _render_figure(
        records,
        output_stem=output_stem,
        formats=parameters.formats,
        dpi=parameters.dpi,
    )
    separate_figure_paths = _render_separate_figures(
        records,
        output_stem=output_stem,
        formats=parameters.formats,
        dpi=parameters.dpi,
    )
    result = {
        "task": "coloring2graph",
        "checkpoint": str(parameters.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": parameters.split,
        "data_dir": str(data_parameters.data_dir.resolve()),
        "device": str(device),
        "presence_threshold": threshold,
        "selection": parameters.selection,
        "indices": selected_indices,
        "figures": [str(path.resolve()) for path in figure_paths],
        "separate_figures": [
            str(path.resolve()) for path in separate_figure_paths
        ],
        "examples": manifest_records,
        "parameters": {
            **asdict(parameters),
            "checkpoint": str(parameters.checkpoint),
            "output_dir": str(parameters.output_dir),
            "data_dir": str(parameters.data_dir) if parameters.data_dir else None,
        },
    }
    manifest_path = output_stem.with_suffix(".json")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["manifest"] = str(manifest_path.resolve())
    return result


def main(arguments: list[str] | None = None) -> None:
    result = export_coloring_examples(parse_parameters(arguments))
    print(f"Exported {len(result['examples'])} fixed {result['split']} examples.")
    for path in result["figures"]:
        print(f"Figure: {path}")
    for path in result.get("separate_figures", []):
        print(f"Separate figure: {path}")
    print(f"Metadata: {result['manifest']}")


if __name__ == "__main__":
    main()
