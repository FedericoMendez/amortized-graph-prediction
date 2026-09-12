"""Compare learned-matcher and legacy-solver losses across saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from any2graph_v2.checkpointing import load_checkpoint
from any2graph_v2.data import build_datamodule
from any2graph_v2.parameter import DataParameters, SolverParameters
from any2graph_v2.solvers.mirror import MirrorSolver
from any2graph_v2.train_cli import build_system


def argument_parser() -> argparse.ArgumentParser:
    """Create the runtime-only command line for this experiment."""

    parser = argparse.ArgumentParser(
        description="Evaluate learned matcher and mirror-descent solver losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True, help="training run directory name")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="checkpoint directory; defaults to the run's all_checkpoints, then checkpoints")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1_000,
        help="number of randomly selected training examples to compare",
    )
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="gpu")
    defaults = SolverParameters(solver_type="mirror", solver_backend="gpu")
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=[defaults.tau],
        help="one or more mirror-descent temperatures to evaluate",
    )
    parser.add_argument("--max-iter-inner", type=int, default=defaults.max_iter_inner)
    parser.add_argument("--tol-inner", type=float, default=defaults.tol_inner)
    parser.add_argument("--max-iter-outer", type=int, default=defaults.max_iter_outer)
    parser.add_argument("--tol-outer", type=float, default=defaults.tol_outer)
    return parser


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate runtime parameters."""

    parser = argument_parser()
    values = parser.parse_args(arguments)
    for name in ("batch_size", "n_samples", "max_iter_inner", "max_iter_outer"):
        value = getattr(values, name)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if values.num_workers is not None and values.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if any(tau <= 0 for tau in values.tau):
        parser.error("every tau value must be positive")
    if len(set(values.tau)) != len(values.tau):
        parser.error("tau values must be distinct")
    for name in ("tol_inner", "tol_outer"):
        if getattr(values, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return values


def _device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--accelerator gpu was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.ckpt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {directory}.")
    return paths


def _data_parameters(stored: DataParameters, values: argparse.Namespace) -> DataParameters:
    overrides = {
        name: value
        for name, value in {
            "data_dir": values.data_dir,
            "data_file": values.data_file,
            "batch_size": values.batch_size,
            "num_workers": values.num_workers,
        }.items()
        if value is not None
    }
    return replace(stored, **overrides)


def _timed_plan(method, device: torch.device) -> tuple[torch.Tensor, float]:
    """Run one matching method and measure its CUDA-synchronized duration."""

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    plan = method()
    torch.cuda.synchronize(device)
    return plan, time.perf_counter() - started


def _results_for_checkpoint(
    checkpoint_path: Path,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mirror_parameters: SolverParameters,
    taus: list[float],
    *,
    checkpoint_index: int,
    checkpoint_count: int,
) -> tuple[dict[str, float | int], dict[float | None, list[float]]]:
    checkpoint, configuration = load_checkpoint(checkpoint_path)
    if configuration.solver.old_solver_approach:
        raise ValueError(
            f"{checkpoint_path} was trained with old_solver_approach and has no learned matcher."
        )
    if configuration.objective.reconstruction_loss != "original":
        raise ValueError(
            "Matcher/mirror comparison requires reconstruction_loss=original; "
            "the mirror solver does not optimize the alternative losses."
        )
    system = build_system(configuration).to(device)
    system.load_state_dict(checkpoint["state_dict"], strict=True)
    system.eval()
    assert system.matcher is not None and system.target_encoder is not None
    matcher_losses: list[torch.Tensor] = []
    solver_losses: dict[float, list[torch.Tensor]] = {tau: [] for tau in taus}
    solvers = {
        tau: MirrorSolver(
            configuration.objective,
            replace(mirror_parameters, tau=tau),
        )
        for tau in taus
    }
    iteration_times: dict[float | None, list[float]] = {
        None: [],
        **{tau: [] for tau in taus},
    }
    with torch.inference_mode():
        for batch in tqdm(
            dataloader,
            desc=f"Checkpoint {checkpoint_index}/{checkpoint_count}",
            unit="batch",
            leave=False,
        ):
            batch = batch.to(device, non_blocking=False)
            prediction = system.predictor.forward_batch(batch)
            matcher_plan, matcher_elapsed = _timed_plan(
                lambda: system.matcher(
                    predicted_node_embeddings=prediction.node_embeddings,
                    target_node_embeddings=system.target_encoder(batch.graphs),
                    target_padding_mask=~batch.graphs.h,
                ).plan,
                device,
            )
            iteration_times[None].append(matcher_elapsed)
            matcher_losses.append(
                system.objective(
                    predicted_graph=prediction.graph_logits,
                    targets=batch.graphs,
                    plan=matcher_plan,
                ).per_graph_loss.cpu()
            )
            for tau in taus:
                solver_plan, solver_elapsed = _timed_plan(
                    lambda: solvers[tau].solve(prediction.graph_logits, batch.graphs).plan,
                    device,
                )
                iteration_times[tau].append(
                    solver_elapsed / solvers[tau].max_iter_outer
                )
                solver_losses[tau].append(
                    system.objective(
                        predicted_graph=prediction.graph_logits,
                        targets=batch.graphs,
                        plan=solver_plan,
                    ).per_graph_loss.cpu()
                )
    matcher_loss = torch.cat(matcher_losses)
    n_steps = int(checkpoint["global_step"])
    loss_row: dict[str, float | int] = {
        "n_steps": n_steps,
        "avg_loss_matcher": float(matcher_loss.mean()),
        "std_loss_matcher": float(matcher_loss.std(unbiased=False)),
    }
    for tau in taus:
        solver_loss = torch.cat(solver_losses[tau])
        suffix = f"{tau:g}"
        loss_row[f"avg_loss_{suffix}"] = float(solver_loss.mean())
        loss_row[f"std_loss_{suffix}"] = float(solver_loss.std(unbiased=False))
    return loss_row, iteration_times


def evaluate(values: argparse.Namespace) -> Path:
    """Evaluate all periodic snapshots and return the written CSV path."""

    run_dir = values.artifacts_dir / values.run_name
    checkpoint_dir = values.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = run_dir / "all_checkpoints"
        if not any(checkpoint_dir.glob("*.ckpt")):
            checkpoint_dir = run_dir / "checkpoints"
    checkpoints = _checkpoint_paths(checkpoint_dir)
    _, first_configuration = load_checkpoint(checkpoints[0])
    data = build_datamodule(_data_parameters(first_configuration.data, values))
    data.setup("fit")
    if data.train_dataset is None:
        raise RuntimeError("The data module did not initialize its training dataset.")
    available = len(data.train_dataset)
    sample_count = min(values.n_samples, available)
    if sample_count == 0:
        raise ValueError("The training dataset contains no examples to sample.")
    indices = torch.randperm(
        available, generator=torch.Generator().manual_seed(first_configuration.data.seed)
    )[:sample_count].tolist()
    dataloader = data._loader(Subset(data.train_dataset, indices), shuffle=False)
    mirror_parameters = SolverParameters(
        solver_type="mirror",
        solver_backend="gpu",
        max_iter_inner=values.max_iter_inner,
        tol_inner=values.tol_inner,
        max_iter_outer=values.max_iter_outer,
        tol_outer=values.tol_outer,
    )
    device = _device(values.accelerator)
    if device.type != "cuda":
        raise RuntimeError("MirrorSolver requires CUDA; pass --accelerator gpu.")
    print(f"Evaluating {sample_count} sampled training examples per checkpoint.")
    loss_rows: list[dict[str, float | int]] = []
    iteration_times: dict[float | None, list[float]] = {
        None: [],
        **{tau: [] for tau in values.tau},
    }
    for index, path in enumerate(
        tqdm(checkpoints, desc="Checkpoints", unit="checkpoint"), start=1
    ):
        checkpoint_loss_row, checkpoint_iteration_times = _results_for_checkpoint(
            path,
            dataloader,
            device,
            mirror_parameters,
            values.tau,
            checkpoint_index=index,
            checkpoint_count=len(checkpoints),
        )
        loss_rows.append(checkpoint_loss_row)
        for tau in (None, *values.tau):
            iteration_times[tau].extend(checkpoint_iteration_times[tau])
    loss_rows.sort(key=lambda row: int(row["n_steps"]))
    output = values.output_csv or run_dir / "matcher_solver_sample_losses.csv"
    time_output = output.with_name("matcher_solver_sample_times.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "n_steps",
                "avg_loss_matcher",
                "std_loss_matcher",
                *[
                    column
                    for tau in values.tau
                    for column in (f"avg_loss_{tau:g}", f"std_loss_{tau:g}")
                ],
            ],
        )
        writer.writeheader()
        writer.writerows(loss_rows)
    with time_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["average_time_per_iter", "std_time_per_iter", "tau"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "average_time_per_iter": float(torch.tensor(times).mean()),
                "std_time_per_iter": float(torch.tensor(times).std(unbiased=False)),
                "tau": "None" if tau is None else tau,
            }
            for tau, times in iteration_times.items()
        )
    print(f"Matcher/mirror-solver sampled losses written to {output}")
    print(f"Matcher/mirror-solver times written to {time_output}")
    return output


def main(arguments: list[str] | None = None) -> None:
    evaluate(parse_arguments(arguments))


if __name__ == "__main__":
    main()
