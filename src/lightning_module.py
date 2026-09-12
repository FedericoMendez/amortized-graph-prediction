"""PyTorch Lightning training system for Any2GraphV2."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor

from any2graph_v2.checkpointing import (
    CHECKPOINT_CONFIGURATION_KEY,
    serialize_experiment_configuration,
)
from any2graph_v2.data.coloring import ColoringGraphBatch
from any2graph_v2.data.fingerprint import FingerprintGraphBatch
from any2graph_v2.data.massspecgym import SpectrumGraphBatch
from any2graph_v2.losses import (
    ReconstructionLossResult,
    build_graph_reconstruction_objective,
    one_hop_feature_diffusion,
)
from any2graph_v2.metrics import (
    GraphMetricResult,
    HardGraphMetrics,
    proper_coloring_rate,
)
from any2graph_v2.models import (
    SpectrumGraphPrediction,
    build_matcher,
    build_predictor,
    build_target_encoder,
)
from any2graph_v2.parameter import (
    DataParameters,
    COLORING_EDGE_CLASSES,
    COLORING_NODE_CLASSES,
    MatcherParameters,
    ModelParameters,
    ObjectiveParameters,
    SolverParameters,
    TargetEncoderParameters,
    TrainingExperimentParameters,
    TrainingParameters,
    VALID_ATOMS_LIST,
    VALID_BOND_TYPES,
)
from any2graph_v2.solvers import FrankWolfeSolver, MirrorSolver, build_solver


def warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    """Return a sample-budget-relative linear-warmup cosine multiplier."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1).")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1].")
    if total_steps == 1:
        return 1.0
    bounded_step = min(max(step, 0), total_steps - 1)
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps > 0 and bounded_step < warmup_steps:
        return (bounded_step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return minimum_ratio
    progress = (bounded_step - warmup_steps) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


class Any2GraphV2LightningModule(pl.LightningModule):
    """Compose prediction, learned matching, loss, and validation metrics."""

    def __init__(
        self,
        *,
        data_parameters: DataParameters | None = None,
        model_parameters: ModelParameters | None = None,
        target_encoder_parameters: TargetEncoderParameters | None = None,
        matcher_parameters: MatcherParameters | None = None,
        solver_parameters: SolverParameters | None = None,
        objective_parameters: ObjectiveParameters | None = None,
        training_parameters: TrainingParameters | None = None,
    ) -> None:
        super().__init__()
        self.data_parameters = data_parameters or DataParameters()
        self.model_parameters = model_parameters or ModelParameters()
        self.target_encoder_parameters = (
            target_encoder_parameters or TargetEncoderParameters()
        )
        self.matcher_parameters = matcher_parameters or MatcherParameters()
        self.solver_parameters = solver_parameters or SolverParameters()
        self.objective_parameters = objective_parameters or ObjectiveParameters()
        self.training_parameters = training_parameters or TrainingParameters()
        if (
            self.solver_parameters.old_solver_approach
            and self.objective_parameters.reconstruction_loss != "original"
        ):
            raise ValueError(
                "Alternative reconstruction losses require the learned matcher; "
                "the detached solvers optimize the original objective."
            )
        self.predictor = build_predictor(
            self.data_parameters,
            self.model_parameters,
            self.target_encoder_parameters,
            feature_diffusion=self.objective_parameters.feature_diffusion,
        )
        if self.data_parameters.task == "coloring2graph":
            n_node_classes = COLORING_NODE_CLASSES
            n_edge_classes = COLORING_EDGE_CLASSES
        else:
            n_node_classes = len(VALID_ATOMS_LIST)
            n_edge_classes = 1 + len(VALID_BOND_TYPES)
        self.old_solver_approach = self.solver_parameters.old_solver_approach
        self.transport_strategy = (
            self.solver_parameters.solver_type
            if self.old_solver_approach
            else f"learned_{self.matcher_parameters.matcher_type}"
        )
        self.target_encoder = None
        self.matcher = None
        self.solver: FrankWolfeSolver | MirrorSolver | None = None
        if self.old_solver_approach:
            self.solver = build_solver(
                self.objective_parameters, self.solver_parameters
            )
        else:
            self.target_encoder = build_target_encoder(
                d_node_decoder=self.model_parameters.d_node_decoder,
                parameters=self.target_encoder_parameters,
                n_node_classes=n_node_classes,
                n_edge_classes=n_edge_classes,
            )
            self.matcher = build_matcher(
                d_node_decoder=self.model_parameters.d_node_decoder,
                n_nodes=self.data_parameters.n_nodes_max,
                parameters=self.matcher_parameters,
                objective_parameters=self.objective_parameters,
            )
        self.objective = build_graph_reconstruction_objective(
            self.objective_parameters
        )
        self.graph_metrics = HardGraphMetrics(
            self.training_parameters.presence_threshold
        )
        self.last_gradient_max = 0.0
        self.last_nonfinite_gradient_tensors = 0
        self.experiment_configuration: TrainingExperimentParameters | None = None
        self._timing_active = False
        self._timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._timing_records: list[dict[str, float | int | str]] = []
        self._timing_epoch_records: list[dict[str, float | int]] = []
        self._timing_epoch_started = 0.0
        self._timing_fit_started = 0.0
        self._timing_epoch_samples = 0
        self._timing_peak_reset = False
        self._latest_solver_iterations: Tensor | None = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Make production checkpoints sufficient for independent evaluation."""

        if self.experiment_configuration is not None:
            checkpoint[CHECKPOINT_CONFIGURATION_KEY] = (
                serialize_experiment_configuration(self.experiment_configuration)
            )

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False
    ):
        """Accept pre-merge MS2Graph checkpoints under their former key prefix."""

        legacy_prefix = "predictor.spectrum_encoder."
        current_prefix = "predictor.input_encoder."
        if any(name.startswith(legacy_prefix) for name in state_dict):
            state_dict = {
                current_prefix + name.removeprefix(legacy_prefix)
                if name.startswith(legacy_prefix)
                else name: value
                for name, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        tokens: Tensor,
        padding_mask: Tensor,
    ):
        return self.predictor(tokens, padding_mask)

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """Move the custom batch and nested dense graph tensors together."""

        del dataloader_idx
        if isinstance(
            batch, (ColoringGraphBatch, FingerprintGraphBatch, SpectrumGraphBatch)
        ):
            return batch.to(device, non_blocking=True)
        return super().transfer_batch_to_device(batch, device, 0)

    def _start_timing_event(self, name: str) -> torch.cuda.Event | None:
        if not self._timing_active:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._timing_events[name] = (start, end)
        return end

    @staticmethod
    def _finish_timing_event(event: torch.cuda.Event | None) -> None:
        if event is not None:
            event.record()

    def _soft_objective(
        self,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
    ) -> tuple[Any, Tensor | None, Tensor, ReconstructionLossResult]:
        prediction = self.predictor.forward_batch(batch)
        plan_total_end = self._start_timing_event("transport_plan_ms")
        self._latest_solver_iterations = None
        if self.old_solver_approach:
            assert self.solver is not None
            target_embeddings = None
            algorithm_end = self._start_timing_event("plan_algorithm_ms")
            with torch.no_grad():
                solver_result = self.solver.solve(
                    prediction.graph_logits, batch.graphs
                )
                plan = solver_result.plan
                self._latest_solver_iterations = solver_result.iterations
            self._finish_timing_event(algorithm_end)
        else:
            assert self.target_encoder is not None and self.matcher is not None
            target_end = self._start_timing_event("target_encoder_ms")
            target_embeddings = self.target_encoder(batch.graphs)
            self._finish_timing_event(target_end)
            target_diffused_features = (
                one_hop_feature_diffusion(batch.graphs)
                if self.objective_parameters.feature_diffusion
                else None
            )
            algorithm_end = self._start_timing_event("plan_algorithm_ms")
            match = self.matcher(
                predicted_node_embeddings=prediction.node_embeddings,
                target_node_embeddings=target_embeddings,
                target_padding_mask=~batch.graphs.h,
                predicted_diffused_features=prediction.graph_logits.nodes.get(
                    "diffused_labels"
                ),
                target_diffused_features=target_diffused_features,
            )
            plan = match.plan
            self._finish_timing_event(algorithm_end)
        self._finish_timing_event(plan_total_end)
        result = self.objective(
            predicted_graph=prediction.graph_logits,
            targets=batch.graphs,
            plan=plan,
        )
        return prediction, target_embeddings, plan, result

    def _log_loss(
        self,
        stage: str,
        result: ReconstructionLossResult,
        batch_size: int,
        *,
        prog_bar: bool,
    ) -> None:
        settings = {
            "on_step": stage == "train",
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": True,
        }
        self.log(f"{stage}/loss", result.loss, prog_bar=prog_bar, **settings)
        for name, value in result.components.items():
            self.log(f"{stage}/{name}", value, **settings)
        for name, value in result.diagnostics.items():
            self.log(f"{stage}/{name}", value, **settings)

    def hard_aligned_prediction(
        self,
        prediction: SpectrumGraphPrediction,
        target_embeddings: Tensor | None,
        plan: Tensor,
        targets: BatchedDenseData,
    ) -> BatchedDenseData:
        """Align predicted slots to target order exactly as validation does."""

        if self.old_solver_approach:
            assert self.solver is not None
            permutations = self.solver.hard_permutations(plan)
        else:
            assert self.matcher is not None and target_embeddings is not None
            hard = self.matcher.hard_match(
                predicted_node_embeddings=prediction.node_embeddings,
                target_node_embeddings=target_embeddings,
                target_padding_mask=~targets.h,
                predicted_diffused_features=prediction.graph_logits.nodes.get(
                    "diffused_labels"
                ),
                target_diffused_features=(
                    one_hop_feature_diffusion(targets)
                    if self.objective_parameters.feature_diffusion
                    else None
                ),
            )
            permutations = hard.permutations
        return prediction.graph_logits.clone().align_(
            permutations.detach().cpu().tolist()
        )

    def training_step(
        self,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        _, _, _, result = self._soft_objective(batch)
        self._log_loss("train", result, len(batch.graphs), prog_bar=True)
        self._timing_epoch_samples += len(batch.graphs)
        if self._latest_solver_iterations is not None:
            self.log(
                "train/solver_iterations_mean",
                self._latest_solver_iterations.float().mean(),
                on_step=True,
                on_epoch=True,
                batch_size=len(batch.graphs),
            )
        return result.loss

    def _evaluation_step(
        self,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
        stage: str,
    ) -> dict[str, Tensor]:
        prediction, target_embeddings, plan, result = self._soft_objective(batch)
        self._log_loss(stage, result, len(batch.graphs), prog_bar=stage == "val")
        aligned = self.hard_aligned_prediction(
            prediction, target_embeddings, plan, batch.graphs
        )
        metrics: GraphMetricResult = self.graph_metrics(aligned, batch.graphs)
        for name, value in metrics.metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch.graphs),
                sync_dist=True,
            )
        if self.data_parameters.task == "coloring2graph":
            coloring_metrics = {
                "color_accuracy": metrics.metrics["atom_accuracy"],
                "edge_accuracy": metrics.metrics["bond_accuracy"],
                "proper_coloring": proper_coloring_rate(
                    aligned, self.training_parameters.presence_threshold
                ),
            }
            for name, value in coloring_metrics.items():
                self.log(
                    f"{stage}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(batch.graphs),
                    sync_dist=True,
                )
        if self._latest_solver_iterations is not None:
            self.log(
                f"{stage}/solver_iterations_mean",
                self._latest_solver_iterations.float().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=len(batch.graphs),
                sync_dist=True,
            )
        return {"loss": result.loss, **metrics.metrics}

    def validation_step(
        self,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
        batch_idx: int,
    ) -> dict[str, Tensor]:
        del batch_idx
        return self._evaluation_step(batch, "val")

    def test_step(
        self,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
        batch_idx: int,
    ) -> dict[str, Tensor]:
        del batch_idx
        return self._evaluation_step(batch, "test")

    def on_train_start(self) -> None:
        if not self.training_parameters.enable_timing_metrics:
            return
        self._timing_fit_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_start(self) -> None:
        if not self.training_parameters.enable_timing_metrics:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._timing_epoch_started = time.perf_counter()
        self._timing_epoch_samples = 0

    def on_train_batch_start(
        self,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        parameters = self.training_parameters
        step = int(self.global_step)
        self._timing_active = bool(
            parameters.enable_timing_metrics
            and self.device.type == "cuda"
            and step >= parameters.timing_warmup_steps
            and (step - parameters.timing_warmup_steps)
            % parameters.timing_interval_steps
            == 0
        )
        self._timing_events = {}
        if (
            parameters.enable_timing_metrics
            and self.device.type == "cuda"
            and not self._timing_peak_reset
            and step >= parameters.timing_warmup_steps
        ):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._timing_peak_reset = True
        self._start_timing_event("train_step_ms")

    def on_before_backward(self, loss: Tensor) -> None:
        del loss
        self._start_timing_event("backward_ms")

    def _log_timing_record(
        self,
        record: dict[str, float | int | str],
        *,
        batch_size: int,
    ) -> None:
        for name in (
            "transport_plan_ms",
            "plan_algorithm_ms",
            "target_encoder_ms",
            "backward_ms",
            "train_step_ms",
            "transport_fraction",
            "samples_per_second",
            "peak_cuda_memory_mib",
        ):
            value = record.get(name)
            if isinstance(value, (float, int)):
                self.log(
                    f"timing/{name}",
                    float(value),
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
        iterations = record.get("solver_iterations_mean")
        if isinstance(iterations, (float, int)):
            self.log(
                "timing/solver_iterations_mean",
                float(iterations),
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: ColoringGraphBatch | FingerprintGraphBatch | SpectrumGraphBatch,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        if not self._timing_active:
            return
        _, step_end = self._timing_events["train_step_ms"]
        step_end.record()
        torch.cuda.synchronize(self.device)
        elapsed = {
            name: start.elapsed_time(end)
            for name, (start, end) in self._timing_events.items()
        }
        batch_size = len(batch.graphs)
        step_ms = elapsed["train_step_ms"]
        plan_ms = elapsed["transport_plan_ms"]
        record: dict[str, float | int | str] = {
            "global_step": int(self.global_step),
            "epoch": int(self.current_epoch),
            "strategy": self.transport_strategy,
            "batch_size": batch_size,
            **elapsed,
            "transport_fraction": plan_ms / max(step_ms, 1e-12),
            "samples_per_second": batch_size * 1_000.0 / max(step_ms, 1e-12),
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(self.device)
            / 2**20,
        }
        if self._latest_solver_iterations is not None:
            record["solver_iterations_mean"] = float(
                self._latest_solver_iterations.float().mean().detach().cpu()
            )
        self._timing_records.append(record)
        self._log_timing_record(record, batch_size=batch_size)
        self._timing_active = False

    def on_train_epoch_end(self) -> None:
        if not self.training_parameters.enable_timing_metrics:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - self._timing_epoch_started
        throughput = self._timing_epoch_samples / max(seconds, 1e-12)
        record = {
            "epoch": int(self.current_epoch),
            "samples": self._timing_epoch_samples,
            "seconds": seconds,
            "samples_per_second": throughput,
        }
        self._timing_epoch_records.append(record)
        self.log("timing/train_epoch_seconds", seconds, on_step=False, on_epoch=True)
        self.log(
            "timing/train_epoch_samples_per_second",
            throughput,
            on_step=False,
            on_epoch=True,
        )

    @staticmethod
    def _timing_summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
        return {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "p90": ordered[p90_index],
            "minimum": ordered[0],
            "maximum": ordered[-1],
        }

    def _write_timing_artifacts(self) -> None:
        if not self.trainer.is_global_zero:
            return
        output_dir = self.training_parameters.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._timing_records:
            names = sorted(
                {name for record in self._timing_records for name in record}
            )
            with (output_dir / "timing_samples.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                writer.writerows(self._timing_records)
        if self._timing_epoch_records:
            with (output_dir / "timing_epochs.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(self._timing_epoch_records[0])
                )
                writer.writeheader()
                writer.writerows(self._timing_epoch_records)

        numeric_names = (
            "transport_plan_ms",
            "plan_algorithm_ms",
            "target_encoder_ms",
            "backward_ms",
            "train_step_ms",
            "transport_fraction",
            "samples_per_second",
            "peak_cuda_memory_mib",
            "solver_iterations_mean",
        )
        summaries = {}
        for name in numeric_names:
            values = [
                float(record[name])
                for record in self._timing_records
                if isinstance(record.get(name), (float, int))
            ]
            if values:
                summaries[name] = self._timing_summary(values)
        final_metrics = {}
        for name, value in self.trainer.callback_metrics.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                final_metrics[name] = float(value.detach().cpu())
            elif isinstance(value, (float, int)):
                final_metrics[name] = float(value)
        payload = {
            "strategy": self.transport_strategy,
            "task": self.data_parameters.task,
            "n_nodes_max": self.data_parameters.n_nodes_max,
            "batch_size": self.data_parameters.batch_size,
            "precision": self.training_parameters.precision,
            "timing_warmup_steps": self.training_parameters.timing_warmup_steps,
            "timing_interval_steps": self.training_parameters.timing_interval_steps,
            "timing_samples": len(self._timing_records),
            "fit_wall_seconds": time.perf_counter() - self._timing_fit_started,
            "metrics": summaries,
            "final_metrics": final_metrics,
            "epochs": self._timing_epoch_records,
        }
        (output_dir / "timing_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    def on_train_end(self) -> None:
        if self.training_parameters.enable_timing_metrics:
            self._write_timing_artifacts()

    def on_after_backward(self) -> None:
        backward_event = self._timing_events.get("backward_ms")
        if backward_event is not None:
            backward_event[1].record()
        gradients = [
            parameter.grad
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            self.last_gradient_max = 0.0
            self.last_nonfinite_gradient_tensors = 0
            return
        finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
        self.last_nonfinite_gradient_tensors = int((~finite).sum().detach().cpu())
        maxima = torch.stack(
            [
                torch.nan_to_num(
                    gradient.detach().abs(),
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=float("inf"),
                ).max()
                for gradient in gradients
            ]
        )
        self.last_gradient_max = float(maxima.max().cpu())
        self.log(
            "train/gradient_max",
            self.last_gradient_max,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/nonfinite_gradient_tensors",
            float(self.last_nonfinite_gradient_tensors),
            on_step=True,
            on_epoch=False,
        )

    def configure_optimizers(self) -> torch.optim.AdamW | dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_parameters.learning_rate,
            weight_decay=self.training_parameters.weight_decay,
        )
        parameters = self.training_parameters
        if parameters.lr_scheduler == "constant":
            return optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)
        minimum_ratio = parameters.min_learning_rate / parameters.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_multiplier(
                step,
                total_steps=total_steps,
                warmup_fraction=parameters.lr_warmup_fraction,
                minimum_ratio=minimum_ratio,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
