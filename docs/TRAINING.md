# Training and validation

## Entry point

The complete model is trained with PyTorch Lightning:

```bash
uv run any2graph-train --help
uv run any2graph-train --wandb-mode online --max-epochs 100
```

All experiment knobs are ordinary `argparse` options with documented defaults.
Stable molecular vocabularies remain constants in `parameter.py`. The CLI uses
`--task` to build either `MassSpecGymDataModule` plus `SpectrumEncoder`, or
`FingerprintGraphDataModule` plus `FingerprintEncoder`. It then builds the
shared `Any2GraphV2LightningModule`, callbacks, and W&B logger from the same
parsed parameter dataclasses.

Named YAML files provide reviewed starting points while retaining the same
argparse interface:

```bash
uv run any2graph-train --config default
uv run any2graph-train --config default --task ms2graph
uv run any2graph-train --config default --scaffold
uv run any2graph-train --config default --matcher-type softsort
```

Every value is written explicitly in each shipped file; optional fields use
YAML `null`. See `configs/README.md` for a parameter-by-parameter tuning guide.

A config is a themed YAML mapping. Its section names are presentation-only;
the loader validates parameter names globally and rejects duplicates. Bare
names resolve under `configs/`, and normal `#` comments document the settings
without entering the parsed data.
Explicit argparse options take precedence, so any value can be changed without
editing the file:

```bash
uv run any2graph-train \
  --config default \
  --data-dir /cluster/data/massspecgym \
  --run-name formula-seed-1 \
  --seed 1
```

Useful bounded smoke-run options are:

```bash
uv run any2graph-train \
  --accelerator cpu \
  --wandb-mode offline \
  --max-epochs 2 \
  --limit-train-batches 2 \
  --limit-val-batches 1
```

`--wandb-mode` is one of `online`, `offline`, or `disabled`. Offline runs retain
the same metric/configuration logging without requiring a network connection.
Outputs are written below `--output-dir` (default `artifacts`). The resolved
arguments are saved as `configuration.json`; local CSV logs are always saved in
`logs/csv/`; W&B files use `wandb/`; and `checkpoints/` contains both the best
validation-loss state and `last.ckpt`.

The resolved configuration is written and W&B/CSV logging is initialized
before dataset setup. The CLI prints flushed `[startup]` messages around
dataset indexing, model construction, logger initialization, and `Trainer.fit`,
including the resolved output path and train/validation dataset sizes.
These markers make a quiet scheduler job diagnosable even when Lightning's
progress bar is disabled. For `ms2graph`, setup reads the spectrum data and
constructs one cached graph per unique molecule; `train_iteration_mode:
spectrum` changes the number of training items but does not reconstruct the
same target graph for each spectrum.

Select the task and model through a YAML in `experiments/paper/configs/`.
Trailing CLI arguments override YAML values.

CLI options use kebab case even when the matching YAML field uses underscores:
for example, YAML `n_nodes_max: 32` is overridden by
`--n-nodes-max 32` (or `--nodes-max 32`), not `--n_nodes_max 32`.

## Resume and checkpoint contract

Resume all Lightning state—model, optimizer, callbacks, epoch, and global
step—with:

```bash
uv run any2graph-train \
  --config default \
  --checkpoint-path artifacts/my-run/checkpoints/last.ckpt \
  --max-epochs 200 \
  --output-dir artifacts/my-run
```

Self-describing checkpoints embed a versioned copy of the complete experiment
configuration. Resume rejects silent changes to model, objective, matcher,
data protocol, batch size, or bounded-batch controls. Device/precision, W&B
mode, worker count, and relocated data directory are accepted as runtime
changes. A constant-rate run may also extend its epoch limit. A warmup-cosine
run may not, because `max_epochs` defines the saved schedule's optimizer-step
budget. The output directory must remain unchanged so Lightning restores one
best-checkpoint ranking across the complete run. This makes a resumed run
explicit without pretending that a scientifically different run is a
continuation.

Legacy checkpoints without experiment metadata cannot be used by the independent evaluator. PyTorch checkpoints can execute Python
deserialization logic; only load trusted files.

## Independent evaluation

The evaluator reconstructs all scientific settings from checkpoint metadata:

```bash
uv run any2graph-evaluate \
  --checkpoint artifacts/my-run/checkpoints/best-099-12345.ckpt \
  --data-dir /cluster/data/massspecgym \
  --fold test \
  --accelerator gpu \
  --output-dir artifacts/my-run/test
```

Only runtime settings and a relocated data path are overridden. `--fold` is
`val` or `test`; test is the default. The command uses the same hard-aligned
metrics as Lightning validation, logs locally to CSV (and optionally W&B), and
writes `<fold>_metrics.json` with the checkpoint path and scalar results.

Do not repeatedly inspect the test fold during model selection. Select models
and thresholds on validation data, then evaluate the test fold once for the
final reported protocol.

## Optimization contract

Every training step follows this differentiable path:

```text
spectrum predictor + target graph encoder
  -> learned soft transport plan T[p,t]
  -> direct-plan reconstruction objective
  -> AdamW update
```

For the solver-based comparison, `--old-solver-approach` replaces the target
encoder and learned matcher with the configured `frank_wolfe` or `mirror`
solve under `torch.no_grad()`. Frank--Wolfe independently selects its CPU or
GPU assignment backend; mirror is CUDA-only and batch-parallel. The resulting
fixed plan enters the same direct-plan loss;
the predictor remains differentiable and the optimizer/loss settings are
unchanged. Target-encoder and matcher options remain present in complete
configs for schema consistency but do not instantiate parameters in this mode.

The default optimizer is AdamW with learning rate `1e-4`, zero weight decay,
constant learning rate, and Lightning gradient-norm clipping at `1.0`. Set
`--lr-scheduler warmup_cosine` to warm up linearly for
`--lr-warmup-fraction` of Lightning's estimated optimizer-step budget and then
decay with a cosine to `--min-learning-rate`. Scheduling occurs after every
optimizer step, and a `LearningRateMonitor` writes the realized rate to CSV and
W&B. The step budget includes `max_epochs` and any bounded-loader settings, so
a scheduled checkpoint must resume with its original maximum epoch count.
Constant-schedule checkpoints may still extend `max_epochs` during resume.

Total loss, every raw objective component, row/column marginal diagnostics,
and gradient-finiteness diagnostics are logged. The objective follows the
corrected GRALE branch: presence BCE, atom CE, categorical bond CE,
derived-adjacency BCE, and marginal KL.

For a fixed data budget, use the same scheduler parameters for every member of
a comparison grid. A fractional warmup maps to the same fraction of examples
when batch size changes, unlike a fixed warmup-step count.

Early stopping and best-checkpoint selection both monitor the epoch-aggregated
`val/loss` in minimization mode. The default patience is ten validation epochs.
This is the differentiable scientific objective; hard reconstruction metrics
remain diagnostics and do not control optimization.

For long epochs, `--validation-interval-minutes 30` adds full validation passes
after approximately 30 minutes of training wall time and still validates at
every epoch end. The timer is checked after each training batch and reset after
validation, so it does not overlap training and the observable interval can be
longer than requested by one training batch plus the validation duration. Each
pass logs the normal `val/*` loss, edit-like distance, exact reconstruction,
and other evaluation metrics to local CSV and the configured W&B run. Extra
passes do not consume early-stopping patience: that callback remains checked
once per completed epoch.

`SpectrumGraphBatch` is a custom dataclass. The Lightning module explicitly
moves its peak tokens, padding mask, collision-energy tensor, and nested
`BatchedDenseData` tensors to the Trainer device, preventing mixed CPU/GPU
batches. Collision-energy conditioning is part of the saved `ModelParameters`,
so checkpoint evaluation reconstructs the same metadata path.

## Hard-aligned validation metrics

Validation first evaluates the same plan-based loss as training. It separately
computes a Hungarian assignment from the learned matcher cost, or projects the
legacy solver plan when `old_solver_approach=true`, hard-aligns the predicted
logits into target order, and reports discrete metrics. Hungarian matching is
nondifferentiable and is never used by the training loss.

The metric conventions are:

- presence errors compare thresholded presence probabilities with target `h`
  over all 48 slots;
- atom errors compare argmax labels only at real target nodes;
- bond errors compare argmax bond/no-bond labels once per unordered pair of
  distinct real target nodes;
- `edit_like = presence_errors + atom_errors + bond_errors`;
- exact reconstruction is one only when all three error counts are zero;
- node-count MAE compares the number of thresholded predicted nodes with the
  target size.

The presence threshold defaults to `0.5` and is configurable. Padding labels
and diagonal self-loops do not affect atom or bond scores. The edit-like score
is an interpretable diagnostic, not a chemically exact graph-edit distance.

## Reproducibility and scope

Lightning, PyTorch DataLoader workers, NumPy, and Python sampling are seeded
from the data `--seed`. Deterministic algorithms are disabled by default to
avoid their performance and operator restrictions; enable `--deterministic`
for controlled reproducibility checks. Exact reproducibility can still depend
on the selected accelerator, precision, PyTorch/CUDA versions, and device
kernels; record the W&B configuration and environment with every scientific
run.

The smoke example is a bounded integration demonstration, not a trained
scientific model or performance result. The training CLI provides execution
machinery; actual hyperparameter selection, final test evaluation,
and multi-seed scientific results must still be run and reported separately.

## Paper experiments

Each YAML in `experiments/paper/configs/` is a standalone training configuration:

```bash
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-fingerprint2mol-matcher.yaml
```

Run the same command inside a GPU allocation on your cluster. Set scheduler
resources according to the selected configuration. See the
[experiment guide](../experiments/paper/README.md) for data setup and config families.

## Comparing matcher and mirror-solver losses across checkpoints

`src/eval_matcher_cli.py` evaluates a fixed, seeded sample of training examples
(default 1,000) at each saved checkpoint, for molecular or Coloring runs:

```bash
PYTHONPATH=src python -m any2graph_v2.eval_matcher_cli \
  --run-name RUN_NAME --tau 0.01 0.1 1 --batch-size 32
```

The script reads `artifacts/RUN_NAME/all_checkpoints/*.ckpt` when present,
otherwise `artifacts/RUN_NAME/checkpoints/*.ckpt`. Use `--checkpoint-dir` for
another directory or `--artifacts-dir` for another run root. Only existing
snapshots can be compared. Training retains a full checkpoint every
`--checkpoint-every` optimizer steps (default 10,000), in addition to best/last
checkpoints. Pass `--checkpoint-every 5000` to save snapshots every 5,000 optimizer steps. The horizontal coordinate `n_steps`
is the saved optimizer step, not the evaluation batch index.

The output `matcher_solver_sample_losses.csv` contains `n_steps`,
`avg_loss_matcher`, `std_loss_matcher`, and `avg_loss_<tau>` / `std_loss_<tau>`
for each requested tau. Standard deviations describe per-example losses.
`matcher_solver_sample_times.csv` contains batch timing statistics; solver
times are divided by the configured outer iteration budget, while matcher
times measure the complete matcher call including target encoding.

Evaluation requires CUDA and a learned-matcher checkpoint with
`reconstruction_loss=original`; the mirror solver cannot optimize the relaxed
alternatives. Feature diffusion follows the checkpoint's objective settings.
The script exports CSV data, without generating a plot. `--n-samples`,
`--data-dir`, `--data-file`, and `--num-workers` provide evaluation overrides.
