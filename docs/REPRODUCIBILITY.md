# Reproducing the code and paper experiments

## Fresh-clone CPU smoke workflow

From a fresh checkout with Python 3.12 and `uv` installed:

```bash
uv sync --frozen
uv run --frozen python scripts/check_paper_manifest.py
uv run --frozen python scripts/reproduce_smoke.py --output-dir artifacts/smoke
```

The smoke run generates 16 synthetic Coloring examples (8 train, 4 validation,
4 test), performs two CPU optimizer steps with the learned matcher, saves full
periodic checkpoints, reloads the second checkpoint, and evaluates a validation
batch. It needs no external data, pretrained weights, Slurm, GPU, or W&B account.
Dependency installation requires network access or a populated package cache.
The locked Linux environment includes CUDA PyTorch even for this CPU workflow;
allow sufficient disk space for its dependencies.

`artifacts/smoke` must not already exist. Choose a new output directory for a
second run. A failed run may leave diagnostic artifacts; successful execution
writes `summary.json` only after training and independent evaluation pass.

Outputs include generated data, `configuration.json`, training checkpoints,
`evaluation/val_metrics.json`, and `summary.json`. The summary records the seed,
optimizer steps, platform, Python/package versions, Git state, source and data
SHA-256 hashes, lockfile hash, checkpoint hash, elapsed workflow time, and metrics.
Git metadata is null in source archives without Git history. Timing is total
smoke workflow time, not a matcher-versus-solver speed benchmark. Absolute paths
and Git status in this local diagnostic should be reviewed before anonymous release.

This is an execution check, not an accuracy/convergence experiment. Fixed seeds
do not guarantee identical numerical results across hardware/library versions.

## Full experimental reproduction

Start with `experiments/paper/manifest.json` and its companion README. The current
entries describe candidate experiment families; none is asserted to reproduce
a finalized paper result. Their launch commands need a configured Slurm cluster
and the task datasets. Each final result must point to the exact resolved config,
seeds, data/split identity, checkpoint, evaluation and plotting commands, and
measured compute. Training and evaluation also have standalone Python CLIs;
Slurm is a convenience for the existing full-scale launchers.

Run `python scripts/check_paper_manifest.py --require-complete` before a paper
release. It intentionally fails until every candidate has a paper identifier
and run evidence. Passing it is not a substitute for reproducing the outputs.

## Validation limits

The CPU smoke workflow checks data generation, training, checkpoint reload, and
validation output. Linux/GPU validation and complete paper runs remain required
before making a full reproducibility claim.

## Verification performed

On 2026-09-11, an isolated local checkout installed with `uv sync --frozen`
completed the self-contained smoke workflow on macOS arm64/Python 3.12.13.
No external datasets, W&B credentials, or Slurm were used. This verifies that
source snapshot on macOS CPU, not Linux/GPU paper-scale reproduction.
