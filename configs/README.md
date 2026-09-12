# Training configuration cheat sheet

Run a preset with:

```bash
source .venv/bin/activate
python -m any2graph_v2.train_cli --config default
```

Explicit command-line options come after `--config` and override YAML values:

```bash
python -m any2graph_v2.train_cli \
  --config default \
  --batch-size 64 \
  --learning-rate 0.00005 \
  --run-name batch64-lr5e5
```

Bare names resolve to `configs/<name>.yaml`; an explicit `.yaml` or `.yml` path
also works. Section names are presentation-only, so related parameters can be
grouped into readable themes such as `hardware`, `predictor`, or
`ms2graph options`. Parameter names are validated globally and must occur once.
`default.yaml` documents every parameter with normal YAML `#` comments.

Command-line booleans always have positive and negative forms, for example
`--use-laplacian-pe` / `--no-use-laplacian-pe` and
`--deterministic` / `--no-deterministic`.

## Presets

| File | Purpose | Important differences |
| --- | --- | --- |
| `default.yaml` | Primary Fingerprint2Graph Sinkhorn run | 4M data, 16 slots, batch 128 |
| `coloring.yaml` | Coloring image-to-graph scaling base | Truncated ResNet18, 64-slot ceiling, FP32, timing enabled |

The scientific presets explicitly use the enlarged widths:

```text
d_token_input = 512
d_node_decoder = 512
d_token_input_feedforward = 1024
d_node_decoder_feedforward = 1024
d_edge_decoder = 512
d_node_target = 256
matcher_dim = 128  # deliberately not doubled
```

The [CPU demo](../scripts/README.md#start-with-the-cpu-demo) defines its tiny
configuration directly and checks training, checkpoint reload, and evaluation.

## Data parameters

| Parameter | Meaning and useful changes |
| --- | --- |
| `--task {coloring2graph,fingerprint2graph,ms2graph}` | Sole router for both dataloader and input encoder. All downstream model and training components are shared. |
| `--data-dir PATH` | MassSpecGym root for `ms2graph` or generated dataset root for `coloring2graph`. |
| `--data-file PATH` | Preprocessed CSV with `smiles` and `fold`. The path is preset-specific; use a PubChem file only after explicitly preprocessing it. |
| `--split-method`, `--train-iteration-mode`, `--eval-iteration-mode`, `--spectrum-representation`, `--n-peaks-max` | MS2Graph loading and sampling controls; ignored by Fingerprint2Graph. |
| `--n-nodes-max N` / `--nodes-max N` | Number of graph slots. Increasing it covers larger molecules but increases matching and bond work, especially the dense `N × N` tensors. All model branches must use the same value. |
| `--n-tokens-max P` | Maximum SOS-plus-active-bit sequence length. Truncation retains the lowest sorted active bit IDs and is recorded in metadata. |
| `--batch-size B` | Examples per optimizer step and GPU. Reduce first if CUDA runs out of memory. Changing it on resume is rejected because it changes optimization. |
| `--num-workers W` | DataLoader worker processes. Increase until loading no longer starves the GPU; do not exceed allocated cluster CPUs. |
| `--pin-memory` / `--no-pin-memory` | Pinned CPU tensors can speed CPU-to-GPU transfer. Enable for GPU training; disable for CPU or debugging. |
| `--scaffold` / `--no-scaffold` | Select Murcko-scaffold or complete-molecule targets for either task. |
| `--remove-invalid-molecules` / `--no-remove-invalid-molecules` | Strict mode removes oversized, unsupported, and empty-scaffold targets. |
| `--seed S` | Seeds sampling, model initialization, and DataLoader workers. Multiple scientific seeds are still required even when deterministic algorithms are disabled. |

## Predictor dimensions and depth

| Parameter | Meaning and scaling effect |
| --- | --- |
| `--d-token-input D` | Fingerprint-token Transformer width. Must be even and divisible by `n-heads`. |
| `--d-node-decoder D` | Predicted-node/query width and target encoder output width. Must be divisible by `n-heads`. This is the main shared latent interface. |
| `--n-heads H` | Attention heads in token and node-query Transformers. Both `d_token_input` and `d_node_decoder` must be divisible by it. More heads do not increase total width by themselves. |
| `--encoder-layers L` | Number of fingerprint-token self-attention layers. |
| `--decoder-layers L` | Number of learned-query self/cross-attention layers. More layers increase token-to-node decoding depth. |
| `--d-token-input-feedforward D` | Hidden MLP width in each input-token Transformer layer. |
| `--d-node-decoder-feedforward D` | Hidden MLP width in each node-decoder Transformer layer. This is independent of the input-token MLP width. |
| `--d-edge-decoder D` | Internal node projection width used by the symmetric categorical bond head. It does not change the five output bond classes. |
| `--dropout P` | Transformer and graph-head dropout probability. `0` is useful for debugging; `0.1` is the current science starting point. |
| `--use-token-positions` / `--no-use-token-positions` | Add fixed sinusoidal positions to token embeddings, matching the reference. Disable only for the set-encoder ablation. |
| `--use-collision-energy` / `--no-use-collision-energy` | Condition MS2Graph peak tokens on collision energy; ignored by Fingerprint2Graph. |
| `--collision-energy-max X` | Sinusoidal collision-energy scale for MS2Graph. |
| `--coloring-image-encoder-type {resnet18,segformer_b1}` | Coloring RGB encoder. `resnet18` is the paper-style truncated ResNet plus spatial Transformer; `segformer_b1` is the native four-stage MiT-B1 multiscale encoder. |
| `--image-feature-grid-size N` | Maximum Coloring encoder output-grid side. The default 16 caps the sequence at 256 tokens; 32 caps it at 1,024 tokens. |

Increasing the representation or feed-forward widths, layer counts, or batch size
raises GPU memory use. `n-nodes-max` is especially important because bond
prediction and reconstruction operate on node pairs.

## Target graph encoder

| Parameter | Meaning and useful changes |
| --- | --- |
| `--target-encoder-type {gnn}` | Relation-specific message-passing encoder. |
| `--d-node-target D` | Hidden width of the bond-aware target GNN before projection to `d_node_decoder`. |
| `--target-layers L` | Relational message-passing layers. More layers expand graph receptive field but may over-smooth representations. |
| `--target-dropout P` | Dropout in target-GNN updates. |
| `--use-laplacian-pe` / `--no-use-laplacian-pe` | Structural Laplacian coordinates are enabled by default to break matching symmetries. Disable for the exact target-permutation-equivariance ablation. |
| `--laplacian-pe-dim K` | Maximum number of nonzero Laplacian eigenvectors appended to atom features. Larger values add structural detail and eigendecomposition/output-projection cost. Repeated eigenspaces remain non-canonical. |

## Matcher

| Parameter | Meaning and useful changes |
| --- | --- |
| `--matcher-type {sinkhorn,softsort}` | `sinkhorn` compares predicted and target embeddings and approximately normalizes both marginals. `softsort` is the cheaper one-sided ablation. |
| `--matcher-dim D` | Projection width used by Sinkhorn's predicted/target cost. It intentionally remains 128 in the enlarged science configs. Softsort does not materially use this field. |
| `--sinkhorn-iterations K` | Update budget for either Sinkhorn mode. More steps improve bistochasticity but increase runtime. It is ignored by Softsort. |
| `--sinkhorn-mode {unrolling,implicit}` | `unrolling` differentiates through a fixed number of unrolled updates. `implicit` uses the custom implicit backward rule after converging the transport plan. |
| `--sinkhorn-tolerance E` | Marginal-error threshold for stopping implicit Sinkhorn iterations. Ignored by unrolling and Softsort modes. |
| `--sinkhorn-check-convergence-every K` | Number of implicit updates between convergence checks. Ignored by unrolling and Softsort modes. |
| `--matcher-epsilon E` | Transport temperature. Smaller values sharpen assignments but can make optimization harder; larger values produce softer plans. |
| `--normalize-matcher-cost` / `--no-normalize-matcher-cost` | Normalize each cost matrix by its sum before applying the temperature. Keep enabled unless running a controlled scale/temperature ablation. |

Short Sinkhorn and Softsort plans are not assumed to be exactly bistochastic;
the objective's marginal KL term handles residual row/column errors.

## Non-learned graph-matching solvers

These parameters live in the dedicated `solver:` YAML section. They are inert
unless `old_solver_approach` is enabled.

| Parameter | Meaning and useful changes |
| --- | --- |
| `--old-solver-approach` / `--no-old-solver-approach` | Omit the target encoder and learned matcher and use the configured non-learned solver. |
| `--solver-type {frank_wolfe,mirror}` | Select the Any2Graph-style Frank–Wolfe solver or the KL-proximal batched CUDA mirror solver. |
| `--solver-backend {cpu,gpu}` | `cpu` transfers solver inputs to float32 CPU tensors and uses SciPy assignment. `gpu` keeps the complete solve on the model CUDA device and uses `torch-linear-assignment`. |
| `--tau A` | Mirror KL-proximal weight. It is the inverse mirror step size: smaller values make sharper, more aggressive updates. Default: `0.1`. |
| `--max-iter-inner K` | Maximum batched Sinkhorn updates inside each mirror step. Default: 100. |
| `--tol-inner E` | Maximum row-marginal error used to stop the inner Sinkhorn solve. Default: `1e-5`. |
| `--max-iter-outer K` | Shared outer-loop budget: Frank–Wolfe linearize/assign/line-search steps or mirror updates. Default: 20. |
| `--tol-outer E` | Shared outer tolerance: absolute objective change for Frank–Wolfe or maximum plan change for mirror. Default: `1e-5`. |

The GPU backend is strict: it raises an error if the batch is not on CUDA or if
`torch-linear-assignment` was built without its CUDA extension. It never
silently falls back to CPU.

The mirror solver ignores `solver_backend`: it is CUDA-only, runs its quadratic
cost, gradient, and Sinkhorn updates batch-parallel on the prediction device,
and needs no native extension or CUDA toolkit. The discrete validation-only
projection still uses the same CPU SciPy assignment as the learned matcher;
this projection is not part of the iterative mirror solve or training loss.

## Objective

The total tracked training and validation objective is:

```text
alpha_presence × presence BCE
+ alpha_atom     × atom CE
+ alpha_bond     × bond CE
+ alpha_adjacency × adjacency BCE
+ alpha_marginal × marginal KL
```

| Parameter | Meaning and useful changes |
| --- | --- |
| `--reconstruction-loss {original,alt_a,alt_b,alt_a_prime,alt_b_prime}` | Select the direct-plan objective or either orientation of the half- or fully-aligned relaxation. Alternatives require the learned matcher. |
| `--alpha-presence A` | Weight for real/padded node-slot presence reconstruction. |
| `--alpha-atom A` | Weight for atom-type reconstruction on real target nodes. |
| `--alpha-bond A` | Weight for categorical bond/no-bond reconstruction on real node pairs. |
| `--alpha-adjacency A` | Weight for GRALE's binary adjacency reconstruction term. Adjacency logits are derived exactly from categorical bond logits, so no extra prediction head is introduced. |
| `--alpha-marginal A` | Weight forcing approximate transport row/column sums toward one. Important for bounded Sinkhorn and Softsort. |
| `--exclude-self-loops` / `--no-exclude-self-loops` | Exclude both predicted and target edge diagonals. Disabled by default to reproduce corrected GRALE; enable only as a molecular ablation. |

At least one alpha must be positive. Raw components are logged separately, so
their magnitudes can be inspected before changing weights.

## Optimization, stopping, device, and logging

| Parameter | Meaning and useful changes |
| --- | --- |
| `--learning-rate LR` | AdamW learning rate. Lower it if loss/gradients are unstable; larger batches may tolerate a larger rate but require validation. |
| `--weight-decay W` | AdamW weight decay. Zero is the current starting point. |
| `--lr-scheduler {constant,warmup_cosine}` | Keep AdamW's rate fixed or apply step-wise linear warmup followed by cosine decay. Constant preserves the historical behavior. |
| `--lr-warmup-fraction F` | Fraction of the estimated optimizer-step budget used for linear warmup. Use the same fraction across batch-size comparisons. |
| `--min-learning-rate LR` | Terminal rate of the cosine schedule. It must not exceed the initial learning rate and is ignored by the constant schedule. |
| `--max-epochs E` | Maximum complete passes through the configured training iterator. It defines the warmup-cosine step budget, so only constant-rate resumes may change it. |
| `--gradient-clip-val G` | Global gradient-norm clipping threshold applied by Lightning. |
| `--early-stopping-patience P` | Stop after this many validation epochs without improvement in epoch-aggregated `val/loss`. |
| `--early-stopping-min-delta D` | Minimum decrease in `val/loss` that counts as improvement. |
| `--presence-threshold P` | Probability threshold used only by discrete validation/test node-presence metrics. It does not change the differentiable training loss. |
| `--accelerator {auto,cpu,gpu}` | Lightning execution backend. Science and smoke configs request `gpu`. |
| `--devices N` | Devices used by Lightning. The shipped SLURM scripts request one GPU, so they use one device. |
| `--precision {32-true,16-mixed,bf16-mixed}` | Numeric precision. `32-true` is safest; mixed precision can reduce memory and improve throughput after a stability check. |
| `--deterministic` / `--no-deterministic` | Deterministic PyTorch algorithms are disabled by default for speed and operator availability. Enable for a controlled reproducibility check; the seed is used either way. |
| `--log-every-n-steps N` | Batch interval for step-level logger writes. Epoch metrics are still aggregated. |
| `--limit-train-batches N` | Optional integer batches per training epoch. Leave unset for science runs; set to one in the smoke test. |
| `--limit-val-batches N` | Optional integer batches per validation epoch. Leave unset for science runs; set to one in the smoke test. |
| `--validation-interval-minutes N` | Run an additional complete validation pass after roughly this much training wall time while retaining validation at every epoch end. The check occurs after the current batch, and a long validation pass can make W&B updates less frequent than the requested interval. |
| `--wandb-mode {online,offline,disabled}` | W&B behavior. Online is the default in every shipped config and SLURM script. Use `offline` only when network access is unavailable; CSV logs are always retained. |
| `--wandb-project NAME` | W&B project name. |
| `--wandb-entity NAME` | Optional W&B user/team. It is intentionally unset in shipped configs so personal ownership is not hard-coded. |
| `--run-name NAME` | Human-readable W&B/experiment name. SLURM train scripts override it with a job-specific name. |
| `--output-dir PATH` | Stores resolved configuration, CSV/W&B logs, best checkpoint, and `last.ckpt`. Resume must reuse the original directory. |
| `--enable-progress-bar` / `--no-enable-progress-bar` | Interactive Lightning progress display. Disable in scheduler logs to avoid control-character noise. |
| `--checkpoint-path PATH` | Resume model, optimizer, callback, epoch, and global-step state. Intentionally unset for fresh configs. |
| `--enable-timing-metrics` / `--no-enable-timing-metrics` | Sample synchronized CUDA plan/full-step timings and persist raw/summary artifacts. Disabled in general presets and enabled by `coloring.yaml`. |
| `--timing-warmup-steps N` | Number of optimizer steps excluded before timing samples and peak-memory measurement. |
| `--timing-interval-steps N` | Optimizer-step interval between synchronized CUDA timing samples. Epoch wall-time throughput is measured separately. |

## Common experiments

Reduce memory pressure:

```text
--batch-size 64
--d-token-input 192 --d-node-decoder 192 \
--d-token-input-feedforward 384 --d-node-decoder-feedforward 384
```

Matcher ablation:

```text
--matcher-type softsort
```

Laplacian ablation:

```text
--no-use-laplacian-pe
```

Reproducibility check:

```text
--deterministic --seed 0
```

Mixed-precision throughput check:

```text
--precision bf16-mixed
```

Change one scientific factor at a time, use a new `run-name` and `output-dir`,
and retain the resolved `configuration.json` with every result.
