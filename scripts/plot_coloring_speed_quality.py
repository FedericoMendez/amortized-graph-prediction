#!/usr/bin/env python3
"""Build paper-ready N20 Coloring speed/quality point-cloud figures.

The script reads local Coloring training artifacts.
Performance is taken from the validation event with the lowest ``val/loss``
(the checkpoint-selection criterion), while speed is the median sampled CUDA
time for the plan algorithm over the complete run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


METHOD_ORDER = ("mirror", "learned_kl", "learned_no_kl")
METHOD_STYLES = {
    "mirror": {
        "label": "Mirror solver",
        "color": "#D1495B",
        "marker": "o",
    },
    "learned_kl": {
        "label": "Learned matcher + marginal KL",
        "color": "#00798C",
        "marker": "s",
    },
    "learned_no_kl": {
        "label": "Learned matcher without marginal KL",
        "color": "#EDA72F",
        "marker": "^",
    },
}
PERFORMANCE_COLUMNS = ("val/exact_reconstruction", "val/edit_like")
EXPECTED_MIRROR_GRID = {
    (10, 20, 0.1),
    (20, 20, 0.1),
    (50, 20, 0.1),
    (100, 20, 0.1),
    (20, 5, 0.1),
    (20, 10, 0.1),
    (20, 40, 0.1),
    (20, 20, 0.03),
    (20, 20, 0.3),
    (50, 40, 0.03),
}
EXPECTED_LEARNED_GRID = {
    (10, 2e-6),
    (50, 2e-6),
    (10, 5e-5),
    (30, 5e-5),
    (50, 2e-5),
    (100, 1e-5),
    (20, 1e-5),
    (20, 2e-5),
    (100, 5e-6),
    (30, 5e-6),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _latest_metrics_csv(run_dir: Path) -> Path:
    candidates = list((run_dir / "logs").rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No Lightning metrics.csv below {run_dir / 'logs'}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _metric_at_event(
    frame: pd.DataFrame,
    *,
    name: str,
    event: pd.Series,
) -> float:
    direct = pd.to_numeric(pd.Series([event.get(name)]), errors="coerce").iloc[0]
    if pd.notna(direct):
        return float(direct)

    candidates = frame[_numeric(frame, name).notna()]
    for key in ("step", "epoch"):
        event_value = pd.to_numeric(
            pd.Series([event.get(key)]), errors="coerce"
        ).iloc[0]
        if pd.isna(event_value) or key not in candidates:
            continue
        matching = candidates[_numeric(candidates, key) == event_value]
        if not matching.empty:
            return float(_numeric(matching, name).iloc[-1])
    raise ValueError(f"Metric {name!r} is absent at the selected validation event.")


def _best_validation_metrics(metrics_path: Path) -> dict[str, float]:
    frame = pd.read_csv(metrics_path)
    losses = _numeric(frame, "val/loss")
    if not losses.notna().any():
        raise ValueError(f"No completed val/loss event in {metrics_path}")
    best_index = losses.idxmin()
    event = frame.loc[best_index]
    result = {
        "best_val_loss": float(losses.loc[best_index]),
        "best_epoch": float(
            pd.to_numeric(pd.Series([event.get("epoch")]), errors="coerce").iloc[0]
        ),
        "best_step": float(
            pd.to_numeric(pd.Series([event.get("step")]), errors="coerce").iloc[0]
        ),
    }
    for name in PERFORMANCE_COLUMNS:
        result[name] = _metric_at_event(frame, name=name, event=event)
    return result


def _timing_metrics(run_dir: Path) -> dict[str, float]:
    summary_path = run_dir / "timing_summary.json"
    if summary_path.exists():
        summary = _json(summary_path)
        metrics = summary.get("metrics", {})
        plan = metrics.get("plan_algorithm_ms", {})
        transport = metrics.get("transport_plan_ms", {})
        if "median" in plan:
            return {
                "plan_algorithm_ms_median": float(plan["median"]),
                "transport_plan_ms_median": float(transport.get("median", math.nan)),
                "timing_samples": float(summary.get("timing_samples", math.nan)),
                "observed_outer_iterations_median": float(
                    metrics.get("solver_iterations_mean", {}).get("median", math.nan)
                ),
            }

    samples_path = run_dir / "timing_samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Neither a usable timing_summary.json nor timing_samples.csv exists in {run_dir}"
        )
    samples = pd.read_csv(samples_path)
    plan = _numeric(samples, "plan_algorithm_ms").dropna()
    if plan.empty:
        raise ValueError(f"No plan_algorithm_ms samples in {samples_path}")
    transport = _numeric(samples, "transport_plan_ms").dropna()
    outer = _numeric(samples, "solver_iterations_mean").dropna()
    return {
        "plan_algorithm_ms_median": float(plan.median()),
        "transport_plan_ms_median": (
            float(transport.median()) if not transport.empty else math.nan
        ),
        "timing_samples": float(len(plan)),
        "observed_outer_iterations_median": (
            float(outer.median()) if not outer.empty else math.nan
        ),
    }


def _method(configuration: dict[str, Any]) -> str:
    solver = configuration["solver"]
    if bool(solver["old_solver_approach"]):
        if solver["solver_type"] != "mirror":
            raise ValueError("The N20 speed/quality plot supports only mirror runs.")
        return "mirror"
    return (
        "learned_kl"
        if float(configuration["objective"]["alpha_marginal"]) > 0.0
        else "learned_no_kl"
    )


def collect_run(run_dir: Path) -> dict[str, Any]:
    configuration = _json(run_dir / "configuration.json")
    if configuration["data"]["task"] != "coloring2graph":
        raise ValueError(f"Not a Coloring run: {run_dir}")
    if int(configuration["data"]["n_nodes_max"]) != 20:
        raise ValueError(f"Not an N20 run: {run_dir}")

    method = _method(configuration)
    matcher = configuration["matcher"]
    solver = configuration["solver"]
    objective = configuration["objective"]
    timing = _timing_metrics(run_dir)
    performance = _best_validation_metrics(_latest_metrics_csv(run_dir))
    sinkhorn_iterations = int(matcher["sinkhorn_iterations"])
    inner = int(solver["max_iter_inner"])
    outer = int(solver["max_iter_outer"])
    nominal_updates = inner * outer if method == "mirror" else sinkhorn_iterations
    if method == "mirror":
        label = f"i{inner}/o{outer}/tau{float(solver['tau']):g}"
    else:
        label = f"s{sinkhorn_iterations}/eps{float(matcher['matcher_epsilon']):g}"

    return {
        "run_name": run_dir.name,
        "method": method,
        "point_label": label,
        "n_nodes_max": int(configuration["data"]["n_nodes_max"]),
        "batch_size": int(configuration["data"]["batch_size"]),
        "seed": int(configuration["data"]["seed"]),
        "sinkhorn_iterations": sinkhorn_iterations if method != "mirror" else math.nan,
        "matcher_epsilon": (
            float(matcher["matcher_epsilon"]) if method != "mirror" else math.nan
        ),
        "alpha_marginal": float(objective["alpha_marginal"]),
        "max_iter_inner": inner if method == "mirror" else math.nan,
        "max_iter_outer": outer if method == "mirror" else math.nan,
        "tau": float(solver["tau"]) if method == "mirror" else math.nan,
        "nominal_sinkhorn_update_budget": nominal_updates,
        "plan_algorithm_seconds_median": timing["plan_algorithm_ms_median"] / 1000.0,
        "transport_plan_seconds_median": timing["transport_plan_ms_median"] / 1000.0,
        **timing,
        **performance,
    }


def collect_runs(
    artifacts_root: Path,
    run_glob: str,
    *,
    allow_incomplete: bool,
    expected_runs: int,
) -> pd.DataFrame:
    directories = sorted(
        path
        for path in artifacts_root.glob(run_glob)
        if path.is_dir() and (path / "configuration.json").exists()
    )
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for directory in directories:
        try:
            records.append(collect_run(directory))
        except (FileNotFoundError, KeyError, ValueError) as error:
            failures.append(f"{directory.name}: {error}")
    if failures:
        message = "Incomplete or incompatible runs:\n  " + "\n  ".join(failures)
        if not allow_incomplete:
            raise RuntimeError(message)
        print(f"warning: {message}", file=sys.stderr)
    if not records:
        raise RuntimeError(
            f"No complete runs matched {artifacts_root / run_glob}."
        )
    if len(records) != expected_runs and not allow_incomplete:
        raise RuntimeError(
            f"Expected {expected_runs} complete runs but found {len(records)}. "
            "Select an experiment with --run-glob, or pass "
            "--allow-incomplete for a diagnostic plot."
        )
    frame = pd.DataFrame.from_records(records)
    if not allow_incomplete:
        _validate_complete_grid(frame)
    method_rank = {name: index for index, name in enumerate(METHOD_ORDER)}
    frame["_method_rank"] = frame["method"].map(method_rank)
    return frame.sort_values(
        ["_method_rank", "nominal_sinkhorn_update_budget", "point_label"]
    ).drop(columns="_method_rank")


def _validate_complete_grid(frame: pd.DataFrame) -> None:
    if set(frame["n_nodes_max"]) != {20} or set(frame["batch_size"]) != {256}:
        raise RuntimeError("The selected runs do not share the N20/batch-256 protocol.")
    if len(set(frame["seed"])) != 1:
        raise RuntimeError("The selected runs mix seeds; select one array job ID.")
    mirror = frame[frame["method"] == "mirror"]
    mirror_grid = {
        (int(row.max_iter_inner), int(row.max_iter_outer), float(row.tau))
        for row in mirror.itertuples()
    }
    if mirror_grid != EXPECTED_MIRROR_GRID or len(mirror) != 10:
        raise RuntimeError("The selected runs do not contain the expected mirror grid.")
    for method in ("learned_kl", "learned_no_kl"):
        learned = frame[frame["method"] == method]
        learned_grid = {
            (int(row.sinkhorn_iterations), float(row.matcher_epsilon))
            for row in learned.itertuples()
        }
        if learned_grid != EXPECTED_LEARNED_GRID or len(learned) != 10:
            raise RuntimeError(
                f"The selected runs do not contain the expected {method} grid."
            )


def _draw_panel(axis: Any, frame: pd.DataFrame, *, x: str, y: str) -> None:
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        style = METHOD_STYLES[method]
        axis.scatter(
            subset[x],
            subset[y],
            s=58,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            label=style["label"],
            zorder=3,
        )
        for _, point in subset.iterrows():
            axis.annotate(
                point["point_label"],
                (point[x], point[y]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color=style["color"],
            )
    axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def make_figure(frame: pd.DataFrame, output_stem: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Run 'uv sync --frozen' after pulling "
            "the updated lockfile."
        ) from error

    plot_frame = frame.copy()
    plot_frame["exact_reconstruction_percent"] = (
        100.0 * plot_frame["val/exact_reconstruction"]
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)

    panels = (
        (axes[0, 0], "plan_algorithm_seconds_median", "exact_reconstruction_percent"),
        (axes[0, 1], "plan_algorithm_seconds_median", "val/edit_like"),
        (axes[1, 0], "nominal_sinkhorn_update_budget", "exact_reconstruction_percent"),
        (axes[1, 1], "nominal_sinkhorn_update_budget", "val/edit_like"),
    )
    for axis, x, y in panels:
        _draw_panel(axis, plot_frame, x=x, y=y)
        axis.set_xscale("log")
    axes[0, 0].set_ylabel("Exact reconstruction (%)")
    axes[1, 0].set_ylabel("Exact reconstruction (%)")
    axes[0, 1].set_ylabel("Edit-like distance (lower is better)")
    axes[1, 1].set_ylabel("Edit-like distance (lower is better)")
    axes[0, 0].set_xlabel("Median plan-algorithm time (s/batch)")
    axes[0, 1].set_xlabel("Median plan-algorithm time (s/batch)")
    axes[1, 0].set_xlabel("Nominal Sinkhorn-update budget")
    axes[1, 1].set_xlabel("Nominal Sinkhorn-update budget")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Coloring N20: transport-plan cost versus reconstruction quality",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.002,
        "Mirror iteration budget = max inner × max outer; learned labels show Sinkhorn iterations. "
        "Performance is measured at the minimum-validation-loss event.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Coloring N20 plan time/iterations against reconstruction quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--run-glob",
        default="coloring-n20-*",
        help="artifact-directory glob; use the array job ID to select one experiment",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_figures/coloring_n20")
    )
    parser.add_argument("--expected-runs", type=int, default=30)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="plot complete runs even if the expected 20-run grid is incomplete",
    )
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = argument_parser().parse_args(arguments)
    frame = collect_runs(
        args.artifacts_root,
        args.run_glob,
        allow_incomplete=args.allow_incomplete,
        expected_runs=args.expected_runs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "coloring_n20_speed_quality.csv"
    frame.to_csv(csv_path, index=False)
    output_stem = args.output_dir / "coloring_n20_speed_quality"
    make_figure(frame, output_stem)
    print(f"Collected {len(frame)} complete runs.")
    print(f"Table: {csv_path}")
    print(f"Figures: {output_stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
