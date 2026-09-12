# Utilities

Training uses the YAMLs in `experiments/paper/configs/` through the standard CLI:

```bash
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-coloring20-matcher.yaml
```

See the [experiment guide](../experiments/paper/README.md) for data preparation,
configuration families, and overrides. Run commands from the repository root.

| Utility | Purpose |
| --- | --- |
| `build_paper_configs.py` | Generate the paper YAMLs; `--check` validates them without writing. |
| `check_paper_manifest.py` | Validate configuration paths and selected-run records. |
| `reproduce_smoke.py` | Generate tiny Coloring data, train on CPU, reload, and evaluate. |
| `count_datasets.py` | Count molecular training, validation, and test samples. |
| `plot_coloring_speed_quality.py` | Export CSV and figures from validation and timing logs. |
| `prepare_data.sh` | Preprocess molecular fingerprints in a Slurm CPU allocation. |
| `coloring/generate_*.sh` | Generate Coloring datasets in Slurm allocations. |
| `export_selected_qualitative_examples.sh` | Export predictions from selected checkpoints. |

For a quick CPU check:

```bash
uv run --frozen python scripts/reproduce_smoke.py --output-dir artifacts/smoke
```

Data-generation and qualitative-export shell scripts retain their scheduler
settings; adjust resources for your cluster. Python equivalents are documented
in [data loading](../docs/DATA.md), [Coloring](../docs/COLORING.md), and
[qualitative examples](../docs/QUALITATIVE_EXAMPLES.md).
