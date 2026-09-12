# Installation and training guide

See the [project overview](../README.md) for the quickstart and paper results.

## Installation

Any2GraphV2 uses `uv` for its locked environment. The standard installation
includes the learned matcher, the CPU Frank--Wolfe solver, and the batched
PyTorch mirror solver:

```bash
uv sync
```

The mirror solver uses only operations already supplied by CUDA PyTorch. It
does not require `nvcc`, `CUDA_HOME`, or a separately installed CUDA toolkit.
Select it with `old_solver_approach: true` and `solver_type: mirror`.

### Optional Frank--Wolfe GPU backend

Only the `frank_wolfe` solver's optional GPU linear-assignment backend requires
the native `torch-linear-assignment` extension. Having an NVIDIA GPU is not
sufficient to compile that extension. Three
separate components are involved:

1. the NVIDIA driver, which must make `nvidia-smi` work;
2. the CUDA runtime bundled with the project's PyTorch wheel;
3. a local CUDA development toolkit containing `nvcc` and CUDA headers.

The PyTorch wheel supplies item 2 but not item 3. Check all three before
building the assignment extension:

```bash
nvidia-smi
nvcc --version
uv run --no-sync python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

This project currently selects PyTorch built for CUDA 12.6, so the recommended
development toolkit is CUDA 12.6. Install `cuda-toolkit-12-6` using NVIDIA's
[Ubuntu installation instructions](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html#ubuntu),
which installs the toolkit without replacing the driver. Do not use the broad
`cuda-12-6` package merely to obtain `nvcc`, because that package also includes
driver packages.

After installing the toolkit, open a new shell and check its location. The
standard package location is `/usr/local/cuda-12.6`:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc"
nvcc --version
```

Persist those exports in the shell or job environment used to build the
extension. If CUDA is installed elsewhere, set `CUDA_HOME` to the directory
containing `bin/nvcc`, `include/`, and `lib64/`.

An error saying `CUDA_HOME environment variable is not set` means the toolkit
is either not installed or cannot be discovered; it does not mean that PyTorch
or the NVIDIA GPU is absent. Do not set `CUDA_HOME` to a nonexistent directory:
the `test -x` and `nvcc --version` checks above must succeed first.

### Build the optional Frank--Wolfe GPU extension

The first command below installs PyTorch and the remaining locked dependencies
while deliberately skipping `torch-linear-assignment`. This is necessary
because that package imports PyTorch while compiling. The second command then
builds the extension against the installed PyTorch and CUDA toolkit:

```bash
uv sync
FORCE_CUDA=1 uv sync --extra frank-wolfe-gpu
```

If the environment was previously synchronized with the CPU-only extension,
force a rebuild after setting `CUDA_HOME`:

```bash
FORCE_CUDA=1 uv sync --extra frank-wolfe-gpu \
  --reinstall-package torch-linear-assignment
```

Verify which extension was installed:

```bash
uv run python -c \
  "from torch_linear_assignment import _backend; print(_backend.has_cuda())"
```

`False` is expected when the optional extension is absent or CPU-only. It must
print `True` before using `solver_type: frank_wolfe` with
`solver_backend: gpu`. This check is irrelevant to `solver_type: mirror`.
The Frank--Wolfe GPU backend deliberately raises an error instead of allowing
the package to fall back silently to CPU. The project disables build isolation
for this dependency so that it builds against the selected PyTorch installation.

After synchronizing the environment, verify the project:

```bash
uv run python -m any2graph_v2.train_cli --help
```

The main configuration is [configs/default.yaml](../configs/default.yaml). YAML is
used for complete reviewed experiments; command-line arguments override it.
YAML field names use Python style (`n_nodes_max`), while command-line options
use kebab case (`--n-nodes-max 32`, or its alias `--nodes-max 32`).

Run the small CPU demo before submitting a long job:

```bash
uv run --frozen python scripts/reproduce_smoke.py --output-dir artifacts/local-smoke
```

This generates its own tiny dataset and checks training, checkpoint reload, and
evaluation. See [reproducibility](REPRODUCIBILITY.md) for details.

## Data preparation

### Fingerprint2Graph

Raw SMILES must first be filtered and split. With no input argument, the
preprocessor acquires PubChem's official CID--SMILES archive, unpacks it to
`src/data/fingerprint2graph/PUBCHEM_raw.csv`, then filters and splits it. An
existing raw file is reused without a download. The preprocessor writes a
derived CSV with `fold`, `molecule_size`, `molecule_scaffold_size`, and
`graph_valid` columns. The last three columns make future strict runs fast:
they avoid reparsing every SMILES just to build the training index.

```bash
uv run python -m any2graph_v2.preprocess_fingerprints \
  --atom-limit 32 --workers 8
```

Pass a raw CSV/TSV path explicitly to preprocess a local source instead.

Existing preprocessed CSVs are upgraded automatically the first time a strict
run needs these columns. The upgrade is streamed, uses the configured worker
count, reports progress, and atomically replaces the derived CSV only after it
completes. Subsequent runs reuse the columns directly.

### MS2Graph

MassSpecGym data is expected under `src/data/massspecgym` by default. Configure
another location with `data_dir` in YAML or `--data-dir` on the command line.
The loader validates target molecules once during setup and caches their dense
graphs for spectrum-level reuse.

### Coloring image to graph

Use the [Coloring guide](COLORING.md) to generate image/graph pairs. The
[experiment index](../experiments/paper/configs/index.json) includes generation
commands for each paper capacity and sample count.

## Training and cluster jobs

Choose a standalone YAML and run the training CLI, locally on a CUDA machine
or inside your cluster's GPU allocation:

```bash
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-fingerprint2mol-matcher.yaml
```

For MS2Scaffold, use `main-ms2scaffold-matcher.yaml`. Override YAML settings with
CLI arguments, for example `--batch-size 64 --seed 1 --output-dir artifacts/run-seed1`.
See the [experiment guide](../experiments/paper/README.md) for the complete grid.

Startup logs state the selected task, output directory, logger status, data
indexing progress, and the selected train/validation sizes. This is the first
place to look when a cluster job appears quiet.

Every run writes the following under `artifacts/<run-name>/` unless an explicit
`--output-dir` is provided:

```text
configuration.json  resolved experiment settings
logs/csv/           local metrics CSV files
wandb/              W&B run files
checkpoints/        best validation-loss checkpoint and last.ckpt
```

W&B is online by default. Set `--wandb-mode offline` or `--wandb-mode disabled`
when appropriate.

## Architecture

```text
fingerprint tokens OR MS2 peaks + collision energy
  -> task-selected Transformer input encoder
  -> learned graph-node query decoder
  -> presence, atom-label, and categorical bond heads

target SMILES -> RDKit dense molecular graph -> target graph encoder
predicted nodes + target nodes -> learned matcher -> direct-plan GRALE loss
```

For the original Any2Graph-style baseline, set `old_solver_approach: true` in a
copied YAML config or pass `--old-solver-approach`. This removes the target
encoder and learned matcher and obtains the same `T[p,t]` plan contract from a
configured no-gradient solver; the predictor, loss, metrics, and training
pipeline stay unchanged. The dedicated `solver:` config section selects a
Frank--Wolfe solver or the batched CUDA mirror solver. Frank--Wolfe can use
SciPy CPU assignment or the optional `torch-linear-assignment` CUDA backend.

The transport plan is indexed `T[p,t]`: rows are predicted slots and columns
are target positions. Soft training uses this plan directly; validation also
uses a Hungarian hard assignment for interpretable graph metrics.

## Repository map

- [configs/README.md](../configs/README.md): every experiment parameter and tuning guide.
- [docs/FINGERPRINT2GRAPH.md](FINGERPRINT2GRAPH.md): fingerprint task and preprocessing details.
- [docs/DATA.md](DATA.md): data contracts, molecular policies, and batching.
- [docs/TRAINING.md](TRAINING.md): training, checkpoints, evaluation, and logging.
- [Architecture](ARCHITECTURE.md) and [loss definitions](LOSS.md): model internals.
- [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md): reproduction checks and limitations.

Upstream papers and implementations are linked in the
[acknowledgments](../README.md#acknowledgments).
