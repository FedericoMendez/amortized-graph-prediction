"""Export fixed input/prediction/ground-truth examples from any checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from torch import Tensor

from any2graph_v2.checkpointing import load_checkpoint
from any2graph_v2.coloring_examples_cli import (
    ColoringExampleParameters,
    export_coloring_examples,
)
from any2graph_v2.data import build_datamodule
from any2graph_v2.data.dense_data import DenseData
from any2graph_v2.data.fingerprint import collate_fingerprint_graph
from any2graph_v2.data.massspecgym import collate_spectrum_graph
from any2graph_v2.metrics import HardGraphMetrics
from any2graph_v2.parameter import (
    FINGERPRINT_SOS_TOKEN_ID,
    MORGAN_FINGERPRINT_SIZE,
    VALID_ATOMS_LIST,
    VALID_BOND_TYPES,
)
from any2graph_v2.train_cli import build_system


@dataclass(frozen=True)
class QualitativeExampleParameters:
    """Runtime-only settings for deterministic qualitative inference."""

    checkpoint: Path
    output_dir: Path = Path("paper_figures/qualitative_examples")
    data_dir: Path | None = None
    data_file: Path | None = None
    split: str = "test"
    indices: tuple[int, ...] = (0, 1, 2, 3)
    selection: str = "indices"
    num_examples: int = 4
    device: str = "auto"
    presence_threshold: float | None = None
    formats: tuple[str, ...] = ("png", "pdf")
    output_name: str = "qualitative_examples"
    dpi: int = 300
    wandb_mode: str = "disabled"
    wandb_project: str = "any2graph-v2"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None


@dataclass(frozen=True)
class MolecularDisplayGraph:
    """Hard molecular node and bond decisions for rendering."""

    presence: np.ndarray
    atom_labels: np.ndarray
    bond_labels: np.ndarray


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed examples from a self-describing checkpoint as input, "
            "hard-aligned prediction, and ground-truth graph panels"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures/qualitative_examples"),
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--data-file", type=Path)
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
        help="use exact indices or scan deterministically for distinct targets",
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
    parser.add_argument("--output-name", default="qualitative_examples")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="any2graph-v2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    return parser


def parse_parameters(
    arguments: list[str] | None = None,
) -> QualitativeExampleParameters:
    parser = argument_parser()
    values = vars(parser.parse_args(arguments))
    indices = tuple(values["indices"])
    formats = tuple(dict.fromkeys(values["formats"]))
    if not indices or any(index < 0 for index in indices):
        parser.error("indices must contain one or more non-negative values")
    if len(set(indices)) != len(indices):
        parser.error("indices must not contain duplicates")
    threshold = values["presence_threshold"]
    if threshold is not None and not 0.0 <= threshold <= 1.0:
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
    return QualitativeExampleParameters(**values)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _molecular_target_identity(task: str, sample: tuple[Any, ...]) -> Any:
    metadata = sample[2]
    if task == "ms2graph":
        return metadata.get("molecule_index", metadata.get("target_smiles"))
    return metadata.get("target_smiles", metadata.get("smiles"))


def _select_molecular_samples(
    dataset: Any,
    *,
    task: str,
    split: str,
    indices: tuple[int, ...],
    selection: str,
    num_examples: int,
) -> tuple[list[int], list[Any]]:
    if selection == "indices":
        invalid = [index for index in indices if index >= len(dataset)]
        if invalid:
            raise IndexError(
                f"{split} indices {invalid} exceed dataset size {len(dataset)}."
            )
        return list(indices), [dataset[index] for index in indices]

    selected_indices: list[int] = []
    samples: list[Any] = []
    seen: set[Any] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        identity = _molecular_target_identity(task, sample)
        if identity is None:
            raise ValueError(f"{task} sample {index} has no stable target identity.")
        if identity in seen:
            continue
        seen.add(identity)
        selected_indices.append(index)
        samples.append(sample)
        if len(samples) == num_examples:
            return selected_indices, samples
    raise ValueError(
        f"Only {len(samples)} distinct targets were found in the {split} split; "
        f"requested {num_examples}."
    )


def _decode_molecular_graph(
    graph: DenseData, *, presence_threshold: float | None
) -> MolecularDisplayGraph:
    if graph.h.dtype == torch.bool:
        presence = graph.h
    else:
        if presence_threshold is None:
            raise ValueError("Predicted graphs require a presence threshold.")
        presence = graph.h.sigmoid() >= presence_threshold
    atom_labels = graph.nodes.labels.argmax(dim=-1)
    symmetric_bond_logits = 0.5 * (
        graph.edges.labels + graph.edges.labels.transpose(0, 1)
    )
    bond_labels = symmetric_bond_logits.argmax(dim=-1)
    real_pairs = presence.unsqueeze(1) & presence.unsqueeze(0)
    bond_labels = bond_labels.masked_fill(~real_pairs, 0)
    bond_labels.fill_diagonal_(0)
    return MolecularDisplayGraph(
        presence=presence.detach().cpu().numpy().astype(bool, copy=False),
        atom_labels=atom_labels.detach().cpu().numpy(),
        bond_labels=bond_labels.detach().cpu().numpy(),
    )


def _force_directed_positions(adjacency: np.ndarray) -> np.ndarray:
    """Return deterministic readable positions without an extra graph dependency."""

    size = len(adjacency)
    if size < 1:
        raise ValueError("A molecular graph must contain at least one target node.")
    if size == 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False) + np.pi / 2.0
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    spring_length = 1.0 / np.sqrt(size)
    edge_pairs = np.argwhere(np.triu(adjacency, k=1))
    temperature = 0.12
    for iteration in range(180):
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        np.fill_diagonal(distances, np.inf)
        distances = np.where(np.isinf(distances), distances, np.maximum(distances, 1e-9))
        repulsion = (
            deltas
            / distances[..., None]
            * (spring_length**2 / distances)[..., None]
        ).sum(axis=1)
        displacement = repulsion
        for left, right in edge_pairs:
            delta = positions[left] - positions[right]
            distance = max(float(np.linalg.norm(delta)), 1e-6)
            attraction = delta / distance * (distance**2 / spring_length)
            displacement[left] -= attraction
            displacement[right] += attraction
        lengths = np.linalg.norm(displacement, axis=1, keepdims=True)
        positions += displacement / np.maximum(lengths, 1e-9) * np.minimum(
            lengths, temperature
        )
        positions -= positions.mean(axis=0, keepdims=True)
        temperature = 0.12 * (1.0 - (iteration + 1) / 180.0)
    lower = positions.min(axis=0)
    span = np.maximum(positions.max(axis=0) - lower, 1e-9)
    return 0.08 + 0.84 * (positions - lower) / span


def _rdkit_positions(smiles: str | None, graph_size: int) -> np.ndarray | None:
    """Recover target atom coordinates in the graph builder's SMILES order."""

    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() < graph_size:
        return None
    rdDepictor.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    positions = np.asarray(
        [
            [conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y]
            for index in range(graph_size)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(positions).all():
        return None
    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    scale = max(float(np.ptp(positions, axis=0).max()), 1e-9)
    return 0.5 + 0.84 * (positions - center) / scale


def _display_positions(
    target: MolecularDisplayGraph, capacity: int, *, smiles: str | None = None
) -> np.ndarray:
    target_size = int(target.presence.sum())
    positions = np.zeros((capacity, 2), dtype=np.float64)
    adjacency = target.bond_labels[:target_size, :target_size] != 0
    target_positions = _rdkit_positions(smiles, target_size)
    positions[:target_size] = (
        target_positions
        if target_positions is not None
        else _force_directed_positions(adjacency)
    )
    extra_count = capacity - target_size
    for offset in range(extra_count):
        row, column = divmod(offset, 10)
        row_count = min(10, extra_count - row * 10)
        positions[target_size + offset] = (
            (column + 1) / (row_count + 1),
            -0.13 - 0.13 * row,
        )
    return positions


def _bond_segments(
    left: np.ndarray, right: np.ndarray, bond_label: int
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    delta = right - left
    length = max(float(np.linalg.norm(delta)), 1e-9)
    normal = np.asarray([-delta[1], delta[0]]) / length
    if bond_label == 2:
        offsets = (-0.012, 0.012)
        styles = ("-", "-")
    elif bond_label == 3:
        offsets = (-0.018, 0.0, 0.018)
        styles = ("-", "-", "-")
    elif bond_label == 4:
        offsets = (0.0,)
        styles = ("--",)
    else:
        offsets = (0.0,)
        styles = ("-",)
    return [
        (left + normal * offset, right + normal * offset, style)
        for offset, style in zip(offsets, styles, strict=True)
    ]


def _adaptive_marker_area(
    node_count: int, *, base_area: float, minimum_area: float
) -> float:
    """Shrink marker area for dense graphs while preserving small-graph labels."""

    reference_nodes = 8
    density_scale = np.sqrt(reference_nodes / max(node_count, reference_nodes))
    return max(minimum_area, base_area * float(density_scale))


def _draw_molecular_graph(
    axis: Any,
    graph: MolecularDisplayGraph,
    positions: np.ndarray,
    *,
    target_size: int,
    title: str,
    mark_errors: bool,
) -> None:
    atom_palette = {
        "C": "#909090",
        "N": "#3050F8",
        "O": "#FF0D0D",
        "S": "#FFFF30",
        "P": "#FF8000",
        "F": "#90E050",
        "Cl": "#1FF01F",
        "Br": "#A62929",
        "I": "#940094",
    }
    visible_nodes = np.flatnonzero(graph.presence)
    if mark_errors:
        visible_nodes = np.union1d(visible_nodes, np.arange(target_size))
    if not len(visible_nodes):
        visible_nodes = np.arange(target_size)
    visible_positions = positions[visible_nodes]
    present_count = int(graph.presence.sum())
    marker_area = _adaptive_marker_area(
        present_count,
        base_area=180.0,
        minimum_area=65.0,
    )
    marker_scale = np.sqrt(marker_area / 180.0)
    label_size = max(4.8, 7.0 * float(marker_scale))
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
    axis.set_facecolor("#FAFAFA")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    for left, right in np.argwhere(np.triu(graph.bond_labels != 0, k=1)):
        label = int(graph.bond_labels[left, right])
        for start, end, style in _bond_segments(
            positions[left], positions[right], label
        ):
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                linestyle=style,
                color="#4D4D4D",
                linewidth=1.4,
                zorder=1,
            )

    if mark_errors:
        missing = np.flatnonzero(~graph.presence[:target_size])
        if len(missing):
            axis.scatter(
                positions[missing, 0],
                positions[missing, 1],
                marker="x",
                s=max(42.0, 0.55 * marker_area),
                color="#B2182B",
                linewidth=1.8,
                zorder=3,
            )

    for node in np.flatnonzero(graph.presence):
        atom_index = int(graph.atom_labels[node])
        atom = VALID_ATOMS_LIST[atom_index]
        is_extra = node >= target_size
        axis.scatter(
            [positions[node, 0]],
            [positions[node, 1]],
            s=marker_area,
            facecolor=atom_palette.get(atom, "#D9D9D9"),
            edgecolor="#B2182B" if is_extra else "#202020",
            linewidth=2.2 if is_extra else 1.0,
            zorder=3,
        )
        axis.text(
            positions[node, 0],
            positions[node, 1],
            atom,
            ha="center",
            va="center",
            fontsize=label_size,
            fontweight="bold",
            color="white" if atom in {"N", "O", "Br", "I"} else "black",
            zorder=4,
        )


def _draw_spectrum_input(axis: Any, sample: tuple[Any, ...], stored: Any) -> None:
    tokens: Tensor = sample[0][: stored.data.n_peaks_max]
    values = tokens.detach().cpu().numpy()
    mz = values[:, 0]
    if stored.data.spectrum_representation == "annotated":
        mz = mz * 1000.0
    intensity = values[:, 1]
    markerline, stemlines, baseline = axis.stem(
        mz, intensity, linefmt="#2166AC", markerfmt=" ", basefmt="#777777"
    )
    del markerline
    stemlines.set_linewidth(0.8)
    baseline.set_linewidth(0.6)
    axis.set_xlabel("m/z", fontsize=8)
    axis.set_ylabel("Intensity", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_title("Input", fontsize=10, pad=7)


def _draw_fingerprint_input(axis: Any, sample: tuple[Any, ...], stored: Any) -> None:
    tokens: Tensor = sample[0]
    values = tokens.detach().cpu().numpy()
    bit_ids = values[values != FINGERPRINT_SOS_TOKEN_ID]
    bit_ids = bit_ids[(bit_ids >= 0) & (bit_ids < MORGAN_FINGERPRINT_SIZE)]
    fingerprint = np.zeros(MORGAN_FINGERPRINT_SIZE, dtype=np.float32)
    fingerprint[bit_ids] = 1.0
    axis.imshow(
        fingerprint.reshape(32, 64),
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    axis.set_xlabel("ECFP4 bit column", fontsize=8)
    axis.set_ylabel("bit row", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_title("Input", fontsize=10, pad=7)


def _render_molecular_figure(
    records: list[dict[str, Any]],
    *,
    task: str,
    stored: Any,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError(
            "Qualitative export requires matplotlib. Run 'uv sync --frozen'."
        ) from error

    figure, axes = plt.subplots(
        len(records), 3, figsize=(11.5, 3.25 * len(records)), squeeze=False
    )
    for row, record in enumerate(records):
        input_axis, prediction_axis, target_axis = axes[row]
        if task == "ms2graph":
            _draw_spectrum_input(input_axis, record["sample"], stored)
        else:
            _draw_fingerprint_input(input_axis, record["sample"], stored)
        input_axis.set_title("Input", fontsize=10, pad=7)
        _draw_molecular_graph(
            prediction_axis,
            record["prediction"],
            record["positions"],
            target_size=record["target_size"],
            title="Prediction",
            mark_errors=True,
        )
        _draw_molecular_graph(
            target_axis,
            record["target"],
            record["positions"],
            target_size=record["target_size"],
            title="Ground truth",
            mark_errors=False,
        )
    legend = [
        Line2D([], [], color="#4D4D4D", label="single bond"),
        Line2D([], [], color="#4D4D4D", linewidth=2.5, label="multiple bond"),
        Line2D([], [], color="#4D4D4D", linestyle="--", label="aromatic bond"),
        Line2D([], [], marker="x", linestyle="none", color="#B2182B", label="missing node"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#B2182B",
            label="extra node",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 1.0), h_pad=1.2, w_pad=1.0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for format_name in formats:
        path = output_stem.with_suffix(f".{format_name}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def _render_separate_molecular_figures(
    records: list[dict[str, Any]],
    *,
    task: str,
    stored: Any,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    """Export every task input and graph as an independent paper-ready file."""

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
        components = ("input", "prediction", "ground-truth")
        for component in components:
            figure_size = (6.2, 3.8) if component == "input" else (4.8, 4.8)
            figure, axis = plt.subplots(figsize=figure_size)
            if component == "input":
                if task == "ms2graph":
                    _draw_spectrum_input(axis, record["sample"], stored)
                else:
                    _draw_fingerprint_input(axis, record["sample"], stored)
                axis.set_title("Input", fontsize=10, pad=7)
            elif component == "prediction":
                _draw_molecular_graph(
                    axis,
                    record["prediction"],
                    record["positions"],
                    target_size=record["target_size"],
                    title="Prediction",
                    mark_errors=True,
                )
            else:
                _draw_molecular_graph(
                    axis,
                    record["target"],
                    record["positions"],
                    target_size=record["target_size"],
                    title="Ground truth",
                    mark_errors=False,
                )
            figure.tight_layout(pad=0.8)
            for format_name in formats:
                path = output_stem.parent / f"{file_prefix}-{component}.{format_name}"
                figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
                paths.append(path)
            plt.close(figure)
    return paths


def _export_molecular_examples(
    parameters: QualitativeExampleParameters,
    checkpoint: dict[str, Any],
    stored: Any,
) -> dict[str, Any]:
    task = stored.data.task
    if task not in {"ms2graph", "fingerprint2graph"}:
        raise ValueError(f"Unsupported molecular qualitative task: {task!r}.")
    example_count = (
        len(parameters.indices)
        if parameters.selection == "indices"
        else parameters.num_examples
    )
    data_overrides: dict[str, Any] = {
        "batch_size": example_count,
        "num_workers": 0,
        "pin_memory": False,
    }
    if parameters.data_dir is not None:
        data_overrides["data_dir"] = parameters.data_dir
    if parameters.data_file is not None:
        data_overrides["data_file"] = parameters.data_file
    data_parameters = replace(stored.data, **data_overrides)
    data = build_datamodule(data_parameters)
    data.setup("test")
    dataset = data.val_dataset if parameters.split == "val" else data.test_dataset
    if dataset is None:
        raise RuntimeError(f"The {parameters.split} dataset was not initialized.")
    selected_indices, samples = _select_molecular_samples(
        dataset,
        task=task,
        split=parameters.split,
        indices=parameters.indices,
        selection=parameters.selection,
        num_examples=parameters.num_examples,
    )
    if task == "ms2graph":
        batch = collate_spectrum_graph(
            samples,
            n_nodes_max=data_parameters.n_nodes_max,
            n_peaks_max=data_parameters.n_peaks_max,
        )
    else:
        batch = collate_fingerprint_graph(
            samples, n_nodes_max=data_parameters.n_nodes_max
        )

    configuration = replace(stored, data=data_parameters)
    system = build_system(configuration)
    system.load_state_dict(checkpoint["state_dict"], strict=True)
    device = _resolve_device(parameters.device)
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
    for batch_index, (sample, dataset_index) in enumerate(
        zip(samples, selected_indices, strict=True)
    ):
        prediction_graph = _decode_molecular_graph(
            aligned[batch_index], presence_threshold=threshold
        )
        target_graph = _decode_molecular_graph(
            batch.graphs[batch_index], presence_threshold=None
        )
        target_size = int(target_graph.presence.sum())
        predicted_size = int(prediction_graph.presence.sum())
        per_graph_metrics = {
            name: float(values[batch_index].detach().cpu())
            for name, values in metric_result.per_graph.items()
        }
        metadata = dict(sample[2])
        target_smiles = metadata.get("target_smiles", metadata.get("smiles"))
        record = {
            "sample": sample,
            "dataset_index": dataset_index,
            "prediction": prediction_graph,
            "target": target_graph,
            "positions": _display_positions(
                target_graph,
                data_parameters.n_nodes_max,
                smiles=str(target_smiles) if target_smiles is not None else None,
            ),
            "target_size": target_size,
            "predicted_size": predicted_size,
            "metrics": per_graph_metrics,
            "metadata": metadata,
        }
        records.append(record)
        manifest_records.append(
            {
                "dataset_index": dataset_index,
                "target_identity": str(_molecular_target_identity(task, sample)),
                "target_size": target_size,
                "predicted_size": predicted_size,
                "smiles": metadata.get("smiles"),
                "target_smiles": metadata.get("target_smiles"),
                "spectrum_index": metadata.get("spectrum_index"),
                "molecule_index": metadata.get("molecule_index"),
                "fold_index": metadata.get("fold_index"),
                "metrics": per_graph_metrics,
            }
        )

    output_stem = parameters.output_dir / parameters.output_name
    figure_paths = _render_molecular_figure(
        records,
        task=task,
        stored=stored,
        output_stem=output_stem,
        formats=parameters.formats,
        dpi=parameters.dpi,
    )
    separate_figure_paths = _render_separate_molecular_figures(
        records,
        task=task,
        stored=stored,
        output_stem=output_stem,
        formats=parameters.formats,
        dpi=parameters.dpi,
    )
    result = {
        "task": task,
        "checkpoint": str(parameters.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": parameters.split,
        "data_dir": str(data_parameters.data_dir.resolve()),
        "data_file": str(data_parameters.data_file.resolve()),
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
            "data_file": str(parameters.data_file) if parameters.data_file else None,
        },
    }
    manifest_path = output_stem.with_suffix(".json")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["manifest"] = str(manifest_path.resolve())
    return result


def _wandb_artifact_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return sanitized or "qualitative-examples"


_SEPARATE_FIGURE_PATTERN = re.compile(
    r".*-example-(?P<ordinal>\d+)-index-(?P<index>\d+)-"
    r"(?P<component>input|prediction|ground-truth)$"
)


def _separate_figure_identity(path: Path) -> tuple[int, int, str] | None:
    """Parse the shared example identity and component from an export filename."""

    match = _SEPARATE_FIGURE_PATTERN.fullmatch(path.stem)
    if match is None:
        return None
    return (
        int(match.group("ordinal")),
        int(match.group("index")),
        match.group("component"),
    )


def _group_separate_pngs(
    paths: list[Path],
) -> list[tuple[str, int, dict[str, Path]]]:
    """Group the three PNG components that belong to each source example."""

    grouped: dict[tuple[int, int], dict[str, Path]] = {}
    for path in paths:
        identity = _separate_figure_identity(path)
        if identity is None:
            raise RuntimeError(f"Unexpected qualitative component filename: {path.name}")
        ordinal, dataset_index, component = identity
        components = grouped.setdefault((ordinal, dataset_index), {})
        if component in components:
            raise RuntimeError(
                f"Duplicate {component} PNG for qualitative example {ordinal}."
            )
        components[component] = path

    required = {"input", "prediction", "ground-truth"}
    result = []
    for (ordinal, dataset_index), components in sorted(grouped.items()):
        missing = required - components.keys()
        if missing:
            raise RuntimeError(
                f"Qualitative example {ordinal} is missing components: "
                + ", ".join(sorted(missing))
            )
        example_id = f"example-{ordinal:02d}-index-{dataset_index}"
        result.append((example_id, dataset_index, components))
    return result


def _log_to_wandb(
    result: dict[str, Any], parameters: QualitativeExampleParameters
) -> None:
    """Log the rendered panel, per-example metrics, and provenance artifact."""

    if parameters.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B qualitative logging requires wandb. Run 'uv sync --frozen'."
        ) from error

    run_name = parameters.wandb_run_name or (
        f"{parameters.checkpoint.parent.parent.name}-qualitative"
    )
    run = wandb.init(
        project=parameters.wandb_project,
        entity=parameters.wandb_entity,
        name=run_name,
        group=parameters.wandb_group,
        job_type="qualitative-evaluation",
        mode=parameters.wandb_mode,
        config={
            "source_checkpoint": result["checkpoint"],
            "checkpoint_epoch": result["checkpoint_epoch"],
            "task": result["task"],
            "split": result["split"],
            "selection": result["selection"],
            "indices": result["indices"],
            "presence_threshold": result["presence_threshold"],
        },
    )
    if run is None:
        raise RuntimeError("wandb.init did not return a run.")
    try:
        panel_paths = [
            Path(path) for path in result["figures"] if Path(path).suffix == ".png"
        ]
        separate_paths = [
            Path(path)
            for path in result.get("separate_figures", [])
            if Path(path).suffix == ".png"
        ]
        if not panel_paths and not separate_paths:
            raise RuntimeError("W&B logging requires PNG among --formats.")
        if panel_paths:
            run.log(
                {
                    "qualitative/panels": [
                        wandb.Image(
                            str(path),
                            caption=(
                                f"{result['task']} {result['split']} indices "
                                f"{result['indices']}"
                            ),
                        )
                        for path in panel_paths
                    ]
                }
            )
        grouped_pngs = _group_separate_pngs(separate_paths) if separate_paths else []
        if grouped_pngs:
            triplet_table = wandb.Table(
                columns=[
                    "example_id",
                    "dataset_index",
                    "Input",
                    "Prediction",
                    "Ground truth",
                ]
            )
            for example_id, dataset_index, components in grouped_pngs:
                triplet_table.add_data(
                    example_id,
                    dataset_index,
                    wandb.Image(
                        str(components["input"]), caption=f"{example_id} — Input"
                    ),
                    wandb.Image(
                        str(components["prediction"]),
                        caption=f"{example_id} — Prediction",
                    ),
                    wandb.Image(
                        str(components["ground-truth"]),
                        caption=f"{example_id} — Ground truth",
                    ),
                )
            run.log(
                {"qualitative/example_triplets": triplet_table}
            )
        table = wandb.Table(
            columns=[
                "dataset_index",
                "target_identity",
                "target_size",
                "predicted_size",
                "edit_like",
                "exact_reconstruction",
                "smiles",
            ]
        )
        for example in result["examples"]:
            metrics = example["metrics"]
            table.add_data(
                example["dataset_index"],
                example["target_identity"],
                example["target_size"],
                example["predicted_size"],
                metrics["edit_like"],
                metrics["exact_reconstruction"],
                example.get("target_smiles") or example.get("smiles"),
            )
        run.log({"qualitative/example_metrics": table})

        artifact = wandb.Artifact(
            name=f"{_wandb_artifact_name(run_name)}-files",
            type="qualitative-examples",
            metadata={
                "source_checkpoint": result["checkpoint"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "task": result["task"],
                "split": result["split"],
                "selection": result["selection"],
                "indices": result["indices"],
            },
        )
        for path_value in [
            *result["figures"],
            *result.get("separate_figures", []),
            result["manifest"],
        ]:
            path = Path(path_value)
            identity = _separate_figure_identity(path)
            if identity is not None:
                ordinal, dataset_index, component = identity
                example_id = f"example-{ordinal:02d}-index-{dataset_index}"
                artifact_name = (
                    f"examples/{example_id}/{component}{path.suffix.lower()}"
                )
            elif path_value == result["manifest"]:
                artifact_name = f"metadata/{path.name}"
            else:
                artifact_name = f"overview/{path.name}"
            artifact.add_file(str(path), name=artifact_name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def export_qualitative_examples(
    parameters: QualitativeExampleParameters,
) -> dict[str, Any]:
    """Dispatch qualitative export according to the checkpoint's stored task."""

    checkpoint, stored = load_checkpoint(parameters.checkpoint)
    if stored.data.task == "coloring2graph":
        if parameters.data_file is not None:
            raise ValueError("Coloring qualitative export does not use --data-file.")
        result = export_coloring_examples(
            ColoringExampleParameters(
                checkpoint=parameters.checkpoint,
                output_dir=parameters.output_dir,
                data_dir=parameters.data_dir,
                split=parameters.split,
                indices=parameters.indices,
                selection=parameters.selection,
                num_examples=parameters.num_examples,
                device=parameters.device,
                presence_threshold=parameters.presence_threshold,
                formats=parameters.formats,
                output_name=parameters.output_name,
                dpi=parameters.dpi,
            )
        )
    else:
        result = _export_molecular_examples(parameters, checkpoint, stored)
    _log_to_wandb(result, parameters)
    return result


def main(arguments: list[str] | None = None) -> None:
    result = export_qualitative_examples(parse_parameters(arguments))
    print(
        f"Exported {len(result['examples'])} fixed {result['split']} "
        f"{result['task']} examples."
    )
    for path in result["figures"]:
        print(f"Figure: {path}")
    for path in result.get("separate_figures", []):
        print(f"Separate figure: {path}")
    print(f"Metadata: {result['manifest']}")


if __name__ == "__main__":
    main()
