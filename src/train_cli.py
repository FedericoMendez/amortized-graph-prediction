"""Command-line entry point for Lightning training."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import timedelta

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger

from any2graph_v2.checkpointing import (
    load_checkpoint,
    validate_resume_configuration,
)
from any2graph_v2.data import build_datamodule
from any2graph_v2.lightning_module import Any2GraphV2LightningModule
from any2graph_v2.parameter import (
    TrainingExperimentParameters,
    TrainingParameters,
    parse_training_experiment_parameters,
)


class EpochEndTimedValidation(pl.Callback):
    """Guarantee epoch-end validation in Lightning's wall-clock mode."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if getattr(trainer, "_val_check_time_interval", None) is None:
            return
        epoch_loop = trainer.fit_loop.epoch_loop
        if epoch_loop.batch_progress.is_last_batch:
            # Lightning's time-based predicate is evaluated immediately after
            # this hook. Expiring its timer adds the epoch-end validation pass.
            trainer._last_val_time = float("-inf")


def build_callbacks(parameters: TrainingParameters) -> list[pl.Callback]:
    """Use validation loss consistently for stopping and best-model selection."""

    checkpoint_dir = parameters.output_dir / "checkpoints"
    callbacks: list[pl.Callback] = [
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=parameters.early_stopping_patience,
            min_delta=parameters.early_stopping_min_delta,
            # Extra wall-clock validations are diagnostics. Preserve the
            # configured patience as a count of complete training epochs.
            check_on_train_epoch_end=(
                True if parameters.validation_interval_minutes is not None else None
            ),
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}-{step}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    callbacks.append(ModelCheckpoint(
        dirpath=parameters.output_dir / "all_checkpoints",
        filename="step-{step:08d}",
        every_n_train_steps=parameters.checkpoint_every,
        save_top_k=-1,
        save_last=False,
        auto_insert_metric_name=False,
    ))
    if parameters.validation_interval_minutes is not None:
        callbacks.append(EpochEndTimedValidation())
    return callbacks


def build_logger(parameters: TrainingParameters) -> WandbLogger | bool:
    """Build an explicit online/offline W&B logger, or disable logging."""

    if parameters.wandb_mode == "disabled":
        return False
    return WandbLogger(
        project=parameters.wandb_project,
        entity=parameters.wandb_entity,
        name=parameters.run_name,
        save_dir=str(parameters.output_dir / "wandb"),
        offline=parameters.wandb_mode == "offline",
        log_model=False,
    )


def build_loggers(parameters: TrainingParameters) -> list[Logger]:
    """Always retain local CSV metrics and optionally mirror them to W&B."""

    loggers: list[Logger] = [
        CSVLogger(save_dir=str(parameters.output_dir / "logs"), name="csv")
    ]
    wandb_logger = build_logger(parameters)
    if isinstance(wandb_logger, WandbLogger):
        loggers.append(wandb_logger)
    return loggers


def build_trainer(
    parameters: TrainingParameters,
    *,
    logger: Logger | list[Logger] | bool | None = None,
) -> pl.Trainer:
    """Construct the reproducible Trainer shared by CLI, tests, and demos."""

    callbacks = build_callbacks(parameters)
    if parameters.lr_scheduler != "constant" and logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    trainer_options = {
        "accelerator": parameters.accelerator,
        "devices": parameters.devices,
        "precision": parameters.precision,
        "max_epochs": parameters.max_epochs,
        "deterministic": parameters.deterministic,
        "gradient_clip_val": parameters.gradient_clip_val,
        "log_every_n_steps": parameters.log_every_n_steps,
        "enable_progress_bar": parameters.enable_progress_bar,
        "callbacks": callbacks,
        "logger": build_loggers(parameters) if logger is None else logger,
        "default_root_dir": str(parameters.output_dir),
    }
    if parameters.limit_train_batches is not None:
        trainer_options["limit_train_batches"] = parameters.limit_train_batches
    if parameters.limit_val_batches is not None:
        trainer_options["limit_val_batches"] = parameters.limit_val_batches
    if parameters.validation_interval_minutes is not None:
        trainer_options["val_check_interval"] = timedelta(
            minutes=parameters.validation_interval_minutes
        )
        trainer_options["check_val_every_n_epoch"] = 1
    return pl.Trainer(**trainer_options)


def build_system(
    configuration: TrainingExperimentParameters,
) -> Any2GraphV2LightningModule:
    system = Any2GraphV2LightningModule(
        data_parameters=configuration.data,
        model_parameters=configuration.model,
        target_encoder_parameters=configuration.target_encoder,
        matcher_parameters=configuration.matcher,
        solver_parameters=configuration.solver,
        objective_parameters=configuration.objective,
        training_parameters=configuration.training,
    )
    system.experiment_configuration = configuration
    return system


def main(arguments: list[str] | None = None) -> None:
    configuration = parse_training_experiment_parameters(arguments)
    output_dir = configuration.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = output_dir / "configuration.json"
    configuration_path.write_text(
        json.dumps(asdict(configuration), indent=2, default=str) + "\n"
    )
    iteration = (
        "graph"
        if configuration.data.task == "coloring2graph"
        else configuration.data.train_iteration_mode
    )
    print(
        "[startup] "
        f"task={configuration.data.task} "
        f"iteration={iteration} "
        f"output_dir={output_dir}",
        flush=True,
    )
    checkpoint_path = configuration.training.checkpoint_path
    if checkpoint_path is not None:
        checkpoint, stored_configuration = load_checkpoint(checkpoint_path)
        validate_resume_configuration(configuration, stored_configuration)
        completed_epoch = int(checkpoint.get("epoch", -1))
        if configuration.training.max_epochs <= completed_epoch + 1:
            raise ValueError(
                "max_epochs must exceed the number of epochs already completed "
                f"by the checkpoint ({completed_epoch + 1})."
            )
    pl.seed_everything(configuration.data.seed, workers=True)
    print(
        f"[startup] initializing loggers (wandb={configuration.training.wandb_mode})...",
        flush=True,
    )
    loggers = build_loggers(configuration.training)
    for logger in loggers:
        logger.log_hyperparams(asdict(configuration))
    print("[startup] loggers ready", flush=True)
    started = time.perf_counter()
    print("[startup] building and indexing the datamodule...", flush=True)
    data = build_datamodule(configuration.data)
    data.setup("fit")
    train_size = len(data.train_dataset) if data.train_dataset is not None else 0
    val_size = len(data.val_dataset) if data.val_dataset is not None else 0
    print(
        "[startup] "
        f"datamodule ready in {time.perf_counter() - started:.1f}s "
        f"(train={train_size}, val={val_size})",
        flush=True,
    )
    started = time.perf_counter()
    print("[startup] building the model...", flush=True)
    system = build_system(configuration)
    print(
        f"[startup] model ready in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print("[startup] initializing Lightning Trainer...", flush=True)
    trainer = build_trainer(configuration.training, logger=loggers)
    print("[startup] starting Trainer.fit", flush=True)
    trainer.fit(
        system,
        datamodule=data,
        ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )


if __name__ == "__main__":
    main()
