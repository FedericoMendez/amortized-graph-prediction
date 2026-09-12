# Coloring image-to-graph scaling benchmark

## Purpose

This benchmark compares three transport-plan paths on the Any2Graph Coloring
task while keeping the RGB input, graph decoder, reconstruction loss, seed,
precision, and validation protocol fixed:

- `learned`: target graph encoder plus the learned GRALE Sinkhorn matcher;
- `mirror`: the detached CUDA KL-proximal mirror solver;
- `frank_wolfe`: the detached Any2Graph conditional-gradient solver with its
  batched GPU linear-assignment backend.

The predictor receives only an image. The colored adjacency graph is ground
truth used by the matcher, reconstruction objective, and metrics. It is never
an input to the predictor.

## Dataset construction

For each sample, the generator:

1. samples a node count and that many distinct seed pixels;
2. partitions a square raster by nearest seed under L1 (taxicab) distance;
3. connects two nodes when their regions share a horizontal or vertical pixel
   boundary;
4. finds a proper four-coloring with DSATUR and an exact fallback, then
   randomizes the four numeric color IDs;
5. renders each region using its node's RGB color and applies deterministic
   Gaussian noise when the sample is loaded.

During training only, each loaded image receives a random horizontal flip with
probability 0.5 followed by a uniformly sampled rotation of 0, 90, 180, or 270
degrees. The graph target is invariant to these image-plane symmetries and is
therefore unchanged. Validation, testing, and qualitative export do not apply
this augmentation.

This follows the input/target construction in the Any2Graph paper. The image
side scales linearly with the graph capacity:

| Maximum nodes | Image resolution |
| ---: | ---: |
| 8 | 32 x 32 |
| 16 | 64 x 64 |
| 20 | 80 x 80 |
| 32 | 128 x 128 |
| 64 | 256 x 256 |

The on-disk format stores uint8 node colors, uint16 undirected edges, and a
zlib-compressed uint8 region map per image. Reconstructing RGB and noise in the
DataLoader avoids storing large uncompressed 256 x 256 arrays. Every directory
has an atomic `manifest.json` containing the resolution, palette, noise level,
seed, format version, class counts, size range, and split statistics.

The default image encoder is the paper's truncated ResNet18: the initial
max-pool and last two residual stages are removed. Its spatial features are
projected, receive a two-dimensional sinusoidal position encoding, and enter
the input Transformer. A selectable `segformer_b1` alternative implements the
four-stage MiT-B1 hierarchy with overlapping patch embeddings, spatial-
reduction attention, Mix-FFNs, and multiscale feature fusion. SegFormer already
performs spatial attention, so its fused tokens go directly to the shared
node-query decoder rather than through the ResNet path's extra input
Transformer. Both encoders are trained from scratch; no external pretrained
weights are downloaded.

`image_feature_grid_size` caps the side of the token grid passed to the shared
decoder. The standard default is 16 x 16. This cap is a project scaling
decision, not a setting reported by the Any2Graph or SegFormer papers.

## Paper configurations

Generate the 20-node dataset before running its configurations:

```bash
uv run --frozen python -m any2graph_v2.generate_coloring \
  --output-dir src/data/coloring/image_max20 --min-nodes 5 --max-nodes 20 \
  --train-samples 320000 --val-samples 40000 --test-samples 40000 --seed 0
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-coloring20-matcher.yaml
```

Generation is substantial; the [CPU demo](REPRODUCIBILITY.md) is the quick
installation check. The [experiment index](../experiments/paper/configs/index.json)
provides generation commands for all six graph capacities. Optional Slurm data
jobs remain under `scripts/coloring/generate_*.sh`.

Use `frontier-*.yaml` for the matching-efficiency grid,
`scaling-*.yaml` for capacity/epsilon runs, and `loss-coloring20-*.yaml` for
loss ablations. The [experiment guide](../experiments/paper/README.md) describes
settings and assumptions. Every YAML can be run with the same training CLI;
no experiment-specific shell launcher is required.

## Stored measurements

W&B and the always-on Lightning CSV logger receive validation loss, every loss
component, marginal errors, edit-like distance, exact reconstruction accuracy,
node-count error, color accuracy, edge accuracy, proper-coloring rate, and the
mean solver outer-iteration count for mirror and Frank--Wolfe.

CUDA timings are sampled after 20 warm-up steps and synchronized every tenth
step:

- `transport_plan_ms`: target encoder plus learned matcher, or the complete
  selected non-learned solve;
- `plan_algorithm_ms`: learned matcher alone, mirror solve, or Frank--Wolfe
  solve;
- `target_encoder_ms`: learned path only;
- `backward_ms`: complete loss backward;
- `train_step_ms`, samples/second, transport fraction, and peak CUDA memory;
- epoch seconds and epoch samples/second, including input decoding and waits.

`train_step_ms` includes the common image encoder and graph predictor;
`transport_plan_ms` intentionally begins after prediction. Every run writes
`timing_samples.csv`, `timing_epochs.csv`, and `timing_summary.json` beside its
configuration and checkpoints. Timing artifacts label the strategies
`learned_sinkhorn`, `mirror`, and `frank_wolfe`. Compare them only at the same
node capacity and batch size.

## Plotting run artifacts

After training the frontier configurations:

```bash
uv run --frozen python scripts/plot_coloring_speed_quality.py \
  --artifacts-root artifacts/paper --run-glob 'frontier-*' --allow-incomplete
```

This exports CSV, PNG, PDF, and SVG summaries from the saved validation and
timing logs. `--allow-incomplete` bypasses the plotter's historical 30-run-grid
check; inspect the collected run count against the 14-config frontier index.
The index records which configuration the paper excludes from its plot.

## Qualitative checkpoint examples

Export fixed test examples from a self-describing Coloring checkpoint with:

```bash
.venv/bin/python -m any2graph_v2.coloring_examples_cli \
  --checkpoint artifacts/<RUN_NAME>/checkpoints/<BEST_CHECKPOINT>.ckpt \
  --indices 0 1 2 3 \
  --device cuda \
  --output-dir paper_figures/coloring_examples/<RUN_NAME>
```

Use `--data-dir src/data/coloring/image_max20` if the checkpoint stores a data
path that differs on the evaluation machine. The default output retains the
combined `coloring_examples.png`/PDF overview and also writes one independent
input, prediction, and ground-truth PNG/PDF per example. Each overview row shows
the noisy RGB input, hard-aligned prediction, and target graph at shared
region-centroid positions. Red crosses mark missed target nodes; red-outlined
nodes are surplus predictions. Graph bounds adapt to include every surplus row.
Panel titles are limited to `Input`, `Prediction`, and `Ground truth`. The JSON
file records all figure paths plus the checkpoint, split, fixed indices, node
counts, and per-example errors. Learned-matcher checkpoints can run on CPU for
small exports, but mirror checkpoints require CUDA.
