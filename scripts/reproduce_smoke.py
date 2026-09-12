"""Generate tiny Coloring data, train, reload, and evaluate without external services."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import lightning.pytorch as pl
import torch

from any2graph_v2.checkpointing import serialize_experiment_configuration
from any2graph_v2.data import build_datamodule
from any2graph_v2.evaluate_cli import EvaluationParameters, evaluate
from any2graph_v2.generate_coloring import generate_dataset
from any2graph_v2.parameter import (
    DataParameters, ModelParameters, MatcherParameters, ObjectiveParameters,
    SolverParameters, TargetEncoderParameters, TrainingExperimentParameters, TrainingParameters,
)
from any2graph_v2.train_cli import build_system, build_trainer

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict:
    # Require a new directory so previous artifacts cannot masquerade as this run.
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(1)
    pl.seed_everything(7, workers=True)
    data_root = output / "data"
    generate_dataset(data_root, min_nodes=5, max_nodes=8, train_samples=8,
                     val_samples=4, test_samples=4, seed=7, workers=1)
    config = TrainingExperimentParameters(
        data=DataParameters(task="coloring2graph", data_dir=data_root, n_nodes_max=8,
                            batch_size=2, num_workers=0, seed=7),
        model=ModelParameters(d_token_input=8, d_node_decoder=8, n_heads=2,
            encoder_layers=1, decoder_layers=1, d_token_input_feedforward=16,
            d_node_decoder_feedforward=16, d_edge_decoder=4, dropout=0,
            use_collision_energy=False, use_token_positions=False, image_feature_grid_size=4),
        target_encoder=TargetEncoderParameters(d_node_target=8, target_layers=1,
            target_dropout=0, use_laplacian_pe=False),
        matcher=MatcherParameters(matcher_dim=8, sinkhorn_iterations=3, matcher_epsilon=0.1),
        solver=SolverParameters(), objective=ObjectiveParameters(),
        training=TrainingParameters(output_dir=output / "training", max_epochs=1,
            accelerator="cpu", wandb_mode="disabled", enable_progress_bar=False,
            limit_train_batches=2, limit_val_batches=1, checkpoint_every=1,
            deterministic=True),
    )
    (output / "configuration.json").write_text(json.dumps(
        serialize_experiment_configuration(config), indent=2) + "\n")
    trainer = build_trainer(config.training, logger=False)
    trainer.fit(build_system(config), datamodule=build_datamodule(config.data))
    if trainer.global_step != 2:
        raise RuntimeError("Smoke training did not complete two optimizer steps")
    checkpoint = config.training.output_dir / "all_checkpoints/step-00000002.ckpt"
    result = evaluate(EvaluationParameters(checkpoint=checkpoint, fold="val",
        accelerator="cpu", batch_size=2, num_workers=0, limit_batches=1,
        wandb_mode="disabled", enable_progress_bar=False, output_dir=output / "evaluation"))
    if not result["metrics"] or not all(math.isfinite(v) for v in result["metrics"].values()):
        raise RuntimeError("Evaluation returned missing or nonfinite metrics")
    provenance = {}
    for key, args in {"commit": ["rev-parse", "HEAD"], "working_tree": ["status", "--porcelain"]}.items():
        process = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        provenance[key] = process.stdout.strip() if process.returncode == 0 else None
    summary = {
        "purpose": "pipeline smoke test; not a paper result", "status": "passed",
        "seed": 7, "optimizer_steps": trainer.global_step, "device": "cpu",
        "elapsed_seconds": time.perf_counter() - start,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "lightning", "numpy", "rdkit"]},
        "git": provenance, "lockfile_sha256": sha256(ROOT / "uv.lock"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                          [Path(__file__).resolve(), *sorted((ROOT / "src").rglob("*.py"))]},
        "data_sha256": {str(p.relative_to(data_root)): sha256(p) for p in sorted(data_root.rglob("*")) if p.is_file()},
        "checkpoint_sha256": sha256(checkpoint), "evaluation": result,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Smoke test passed: {output / 'summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
