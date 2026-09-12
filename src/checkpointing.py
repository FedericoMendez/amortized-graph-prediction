"""Self-describing experiment configuration stored in Lightning checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from any2graph_v2.parameter import (
    DataParameters,
    MatcherParameters,
    ModelParameters,
    ObjectiveParameters,
    SolverParameters,
    TargetEncoderParameters,
    TrainingExperimentParameters,
    TrainingParameters,
)

CHECKPOINT_CONFIGURATION_KEY = "any2graph_v2_experiment_configuration"
CHECKPOINT_SCHEMA_VERSION = 15


def _migrate_dimension_names(
    model_values: dict[str, Any], target_values: dict[str, Any]
) -> None:
    """Upgrade schema-1/2 dimension names to the explicit stream/stage scheme."""

    input_feedforward = model_values.pop("feedforward_dim")
    model_values["d_token_input_feedforward"] = input_feedforward
    model_values["d_node_decoder_feedforward"] = input_feedforward
    for old, new in {
        "d_model": "d_token_input",
        "d_node": "d_node_decoder",
        "edge_dim": "d_edge_decoder",
    }.items():
        if old in model_values:
            model_values[new] = model_values.pop(old)
    for old, new in {
        "target_hidden_dim": "d_node_target",
        "target_edge_dim": "d_edge_target",
        "target_global_dim": "d_global_target",
        "target_feedforward_dim": "d_node_target_feedforward",
        "target_edge_feedforward_dim": "d_edge_target_feedforward",
        "target_global_feedforward_dim": "d_global_target_feedforward",
    }.items():
        if old in target_values:
            target_values[new] = target_values.pop(old)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def serialize_experiment_configuration(
    configuration: TrainingExperimentParameters,
) -> dict[str, Any]:
    """Convert validated dataclasses to a versioned, JSON-safe payload."""

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "configuration": _json_safe(asdict(configuration)),
    }


def deserialize_experiment_configuration(
    payload: dict[str, Any],
) -> TrainingExperimentParameters:
    """Reconstruct and revalidate an experiment configuration payload."""

    schema_version = payload.get("schema_version")
    if schema_version not in {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError(
            "Unsupported checkpoint configuration schema: "
            f"{payload.get('schema_version')!r}."
        )
    values = payload.get("configuration")
    if not isinstance(values, dict):
        raise ValueError("Checkpoint configuration payload is missing configuration.")
    try:
        data_values = dict(values["data"])
        if schema_version == 1 and "task" not in data_values:
            # The Fingerprint2Graph branch introduced data_file; the original
            # MS2Graph checkpoints did not contain that field.
            data_values["task"] = (
                "fingerprint2graph" if "data_file" in data_values else "ms2graph"
            )
        model_values = dict(values["model"])
        if schema_version <= 11:
            model_values.setdefault("graph_decoder_type", "node_query")
        if schema_version <= 6:
            model_values.setdefault("image_feature_grid_size", 16)
        if schema_version <= 7:
            model_values.setdefault("coloring_image_encoder_type", "resnet18")
        target_values = dict(values["target_encoder"])
        matcher_values = dict(values["matcher"])
        solver_values = dict(values.get("solver", {}))
        if schema_version <= 3:
            solver_values.setdefault(
                "old_solver_approach",
                matcher_values.pop("old_solver_approach", False),
            )
        legacy_max_iterations = solver_values.pop("solver_max_iterations", None)
        legacy_tolerance = solver_values.pop("solver_tolerance", None)
        if solver_values.get("solver_type", "frank_wolfe") == "frank_wolfe":
            if legacy_max_iterations is not None:
                solver_values["max_iter_outer"] = legacy_max_iterations
            if legacy_tolerance is not None:
                solver_values["tol_outer"] = legacy_tolerance
        else:
            if legacy_max_iterations is not None:
                solver_values.setdefault("max_iter_outer", legacy_max_iterations)
            if legacy_tolerance is not None:
                solver_values.setdefault("tol_outer", legacy_tolerance)
        if schema_version in {1, 2}:
            _migrate_dimension_names(model_values, target_values)
        training_values = dict(values["training"])
        if schema_version <= 8:
            training_values.setdefault("validation_interval_minutes", None)
        if schema_version <= 9:
            training_values.setdefault("lr_scheduler", "constant")
            training_values.setdefault("lr_warmup_fraction", 0.05)
            training_values.setdefault("min_learning_rate", 0.0)
        objective_values = dict(values["objective"])
        if schema_version <= 10:
            objective_values.setdefault("feature_diffusion", False)
            objective_values.setdefault("alpha_feature_diffusion", 1.0)
        if schema_version <= 12:
            objective_values.setdefault("reconstruction_loss", "original")
        if schema_version < 15:
            # Older GNN checkpoints recorded unused alternative-encoder settings.
            for key in (
                "target_heads", "d_edge_target", "d_global_target",
                "d_node_target_feedforward", "d_edge_target_feedforward",
                "d_global_target_feedforward", "target_flex_block_size",
                "target_flex_use_compile",
            ):
                target_values.pop(key, None)
        return TrainingExperimentParameters(
            data=DataParameters(**data_values),
            model=ModelParameters(**model_values),
            target_encoder=TargetEncoderParameters(**target_values),
            matcher=MatcherParameters(**matcher_values),
            solver=SolverParameters(**solver_values),
            objective=ObjectiveParameters(**objective_values),
            training=TrainingParameters(**training_values),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid experiment configuration in checkpoint.") from error


def load_checkpoint(
    checkpoint_path: Path | str,
) -> tuple[dict[str, Any], TrainingExperimentParameters]:
    """Load a trusted Lightning checkpoint and its validated configuration."""

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a mapping.")
    payload = checkpoint.get(CHECKPOINT_CONFIGURATION_KEY)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint {path} lacks the required experiment metadata."
        )
    return checkpoint, deserialize_experiment_configuration(payload)


def validate_resume_configuration(
    requested: TrainingExperimentParameters,
    stored: TrainingExperimentParameters,
) -> None:
    """Reject silent scientific configuration drift when resuming a run."""

    requested_values = _json_safe(asdict(requested))
    stored_values = _json_safe(asdict(stored))
    operational_fields = {
        "data": {"data_dir", "data_file", "num_workers", "pin_memory"},
        "training": {
            "accelerator",
            "devices",
            "precision",
            "log_every_n_steps",
            "checkpoint_every",
            "wandb_mode",
            "wandb_project",
            "wandb_entity",
            "run_name",
            "enable_progress_bar",
            "checkpoint_path",
            "validation_interval_minutes",
        },
    }
    if (
        requested.training.lr_scheduler == "constant"
        and stored.training.lr_scheduler == "constant"
    ):
        operational_fields["training"].add("max_epochs")
    for section, names in operational_fields.items():
        for name in names:
            requested_values[section].pop(name, None)
            stored_values[section].pop(name, None)
    if requested_values != stored_values:
        raise ValueError(
            "Resume configuration changes scientific or data-sampling settings. "
            "Use the original config and override only documented runtime fields."
        )
