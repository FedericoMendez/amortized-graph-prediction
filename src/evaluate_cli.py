"""Independent validation/test evaluation from a self-describing checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import lightning.pytorch as pl

from any2graph_v2.checkpointing import load_checkpoint
from any2graph_v2.data import build_datamodule
from any2graph_v2.parameter import TrainingParameters
from any2graph_v2.train_cli import build_loggers, build_system


@dataclass(frozen=True)
class EvaluationParameters:
    """Runtime-only overrides; scientific model settings come from checkpoint."""

    checkpoint: Path
    fold: str = "test"
    data_dir: Path | None = None
    data_file: Path | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    wandb_mode: str = "disabled"
    wandb_project: str = "any2graph-v2"
    wandb_entity: str | None = None
    run_name: str | None = None
    output_dir: Path = Path("artifacts/evaluation")
    limit_batches: int | None = None
    enable_progress_bar: bool = True


def evaluation_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an Any2GraphV2 checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", choices=("val", "test"), default="test")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("32-true", "16-mixed", "bf16-mixed"),
        default="32-true",
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="any2graph-v2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument(
        "--enable-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_evaluation_parameters(
    arguments: list[str] | None = None,
) -> EvaluationParameters:
    parser = evaluation_argument_parser()
    values = vars(parser.parse_args(arguments))
    for name in ("batch_size", "devices", "limit_batches"):
        value = values[name]
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if values["num_workers"] is not None and values["num_workers"] < 0:
        parser.error("num-workers must be non-negative")
    return EvaluationParameters(**values)


def _runtime_training_parameters(
    stored: TrainingParameters, runtime: EvaluationParameters
) -> TrainingParameters:
    return replace(
        stored,
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        precision=runtime.precision,
        wandb_mode=runtime.wandb_mode,
        wandb_project=runtime.wandb_project,
        wandb_entity=runtime.wandb_entity,
        run_name=runtime.run_name,
        output_dir=runtime.output_dir,
        enable_progress_bar=runtime.enable_progress_bar,
        checkpoint_path=None,
    )


def evaluate(parameters: EvaluationParameters) -> dict[str, Any]:
    """Evaluate one trusted checkpoint and persist a JSON metric record."""

    checkpoint, stored = load_checkpoint(parameters.checkpoint)
    data_parameters = stored.data
    data_overrides: dict[str, Any] = {}
    if parameters.data_dir is not None:
        data_overrides["data_dir"] = parameters.data_dir
    if parameters.data_file is not None:
        data_overrides["data_file"] = parameters.data_file
    if parameters.batch_size is not None:
        data_overrides["batch_size"] = parameters.batch_size
    if parameters.num_workers is not None:
        data_overrides["num_workers"] = parameters.num_workers
    data_parameters = replace(data_parameters, **data_overrides)
    training_parameters = _runtime_training_parameters(stored.training, parameters)
    configuration = replace(
        stored, data=data_parameters, training=training_parameters
    )

    pl.seed_everything(configuration.data.seed, workers=True)
    system = build_system(configuration)
    system.load_state_dict(checkpoint["state_dict"], strict=True)
    data = build_datamodule(configuration.data)
    loggers = build_loggers(training_parameters)
    for logger in loggers:
        logger.log_hyperparams(
            {
                "checkpoint_configuration": asdict(stored),
                "evaluation": asdict(parameters),
            }
        )

    trainer_options: dict[str, Any] = {
        "accelerator": parameters.accelerator,
        "devices": parameters.devices,
        "precision": parameters.precision,
        "deterministic": stored.training.deterministic,
        "logger": loggers,
        "enable_checkpointing": False,
        "enable_progress_bar": parameters.enable_progress_bar,
        "default_root_dir": str(parameters.output_dir),
    }
    if parameters.limit_batches is not None:
        trainer_options[
            "limit_val_batches" if parameters.fold == "val" else "limit_test_batches"
        ] = parameters.limit_batches
    trainer = pl.Trainer(**trainer_options)
    if parameters.fold == "val":
        raw_results = trainer.validate(system, datamodule=data, verbose=True)
    else:
        raw_results = trainer.test(system, datamodule=data, verbose=True)
    if len(raw_results) != 1:
        raise RuntimeError("Expected exactly one evaluation dataloader result.")

    metrics = {name: float(value) for name, value in raw_results[0].items()}
    result = {
        "checkpoint": str(parameters.checkpoint.resolve()),
        "fold": parameters.fold,
        "metrics": metrics,
    }
    parameters.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = parameters.output_dir / f"{parameters.fold}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Evaluation metrics written to {result_path}")
    return result


def main(arguments: list[str] | None = None) -> None:
    evaluate(parse_evaluation_parameters(arguments))


if __name__ == "__main__":
    main()
