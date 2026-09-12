# Paper experiments

Run a configuration with the standard training CLI:

```bash
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-coloring20-matcher.yaml
```

Prepare the datasets first using the [data guide](../../docs/DATA.md) and
[Coloring guide](../../docs/COLORING.md). Runs target one CUDA GPU. Logging to
W&B is disabled by default; add `--wandb-mode online` to enable it.
CLI arguments override YAML values, including data paths, seeds, and output directories.

## Configurations

| Paper experiment | YAML files in `configs/` | Runs |
| --- | --- | ---: |
| Table 1: main comparisons | `main-*.yaml` | 12 |
| Table 2: loss ablations | `loss-*.yaml` | 15 |
| Table 6: matching efficiency | `frontier-*.yaml` | 14 |
| Table 7: matcher capacity/epsilon grid | `scaling-*-matcher-*.yaml` | 18 |
| Figure 2: mirror capacity baseline | `scaling-*-mirror.yaml` | 6 |

Each YAML contains the full configuration and a distinct output directory under
`artifacts/paper/`. [index.json](configs/index.json) maps configurations to CSV
rows and provides six Coloring data-generation commands with the Table 3 sample
counts. The generated datasets use an 80/10/10 train/validation/test split and seed 0.

For example, compare a loss variant with the original objective:

```bash
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/loss-coloring20-alt_a.yaml
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/loss-coloring20-original.yaml
```

## How the tables become configurations

Tables 3–5 set graph capacity, architecture, optimization, and training budgets.
Table 6 supplies solver iterations, epsilon, and marginal-KL settings. Table 7
supplies the scaling epsilon candidates, converted from units of 1e-6. Results
columns such as edit distance and accuracy are never passed to training.

The generator makes the following choices explicit:

- Parameters absent from the tables inherit `configs/default.yaml` or
  `configs/coloring.yaml`. Feed-forward widths are twice the corresponding
  encoder/decoder widths. Molecular feature diffusion follows the molecular launchers.
- Main comparisons and loss ablations use launcher-derived epsilon values:
  `3e-5` for molecular tasks and `3e-5 * (10 / N)^2` for Coloring. These are
  starting configurations; final selected run records are not available.
- Table 4 takes precedence over older launcher budgets: Fingerprint2Mol uses
  10 epochs, and the minimum learning rate is `1e-5`. Mirror main/scaling runs
  use half the training epochs; the efficiency grid retains the full budget.
- Table 7 uses the printed, rounded epsilon values exactly. No selected matcher
  config is generated from the blank Figure 2 epsilon cells; use the candidate grid.
- Table 2 maps `J`, `Ja`, `Jb`, reversed `Ja`, and reversed `Jb` to `original`,
  `alt_a`, `alt_b`, `alt_a_prime`, and `alt_b_prime`. These run the repository's
  implementations; the probability-KL versus cross-entropy distinction and
  transport conventions are described in [LOSS.md](../../docs/LOSS.md).
- Relationformer configs select the implemented relation-token decoder with this
  repository's learned matching and objective. They do not implement a separate
  upstream Relationformer training pipeline. FGWBARY has no implementation here.
- The 100-inner/100-outer mirror configuration is retained; its index entry
  records that the paper excludes it from the plot.

These configurations are executable experiment definitions. Matching the reported
scores also requires the original data versions, seeds, and selected checkpoints.

## Regenerate and validate

```bash
uv run --frozen python scripts/build_paper_configs.py
uv run --frozen python scripts/build_paper_configs.py --check
```

The builder reads the [CSVs](csv/) and validates every YAML through the training
CLI parser. Edit the source tables or builder when changing the experiment grid;
use CLI overrides for individual runs. `--check` detects drift in committed configs.

## Run records

`manifest.json` lists a training command for each of the 65 YAML configurations.
`configs/index.json` maps them to source tables. Record the selected runs before
linking an experiment to a final paper result.

Before release, make one entry per table row or figure series, attach the final runs, and set `paper_result` to the manuscript
identifier. Attach `run_evidence` containing:

- `commit`, `resolved_config`, and `seeds`: exact code and all configurations used.
- `dataset_version`, `dataset_sha256`, and `split_definition`: immutable input identity and filtering/split procedure.
- `checkpoint`, `evaluation_command`, and `metrics`: artifact locations/checksums, full argument lists, and per-seed results.
- `aggregation`: metric definitions, across-seed summary, and uncertainty calculation.
- `hardware`, `runtime_seconds`, and `peak_memory_gb`: measured resources, including the timing scope and warmup policy.
- `figure_or_table_command`: command rebuilding the reported output from stored results.

Then set `status` to `verified` and run:

```bash
python scripts/check_paper_manifest.py --require-complete
```

The completeness check requires the run records listed above. Review the
underlying logs, artifacts, and paper mapping alongside this structural check.
Launcher configs are starting presets; the archived resolved configuration
must include every CLI override and environment-dependent choice. Preserve
validation-based checkpoint selection and keep the final test fold separate.

For a small CPU pipeline check, use the smoke workflow in
[Reproducibility](../../docs/REPRODUCIBILITY.md).
