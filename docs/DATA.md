# Data loading and task routing

`DataParameters.task` is the sole dataloader selector. `ms2graph` builds
`MassSpecGymDataModule`; `fingerprint2graph` builds
`FingerprintGraphDataModule`; `coloring2graph` uses generated Coloring images
(see [Coloring](COLORING.md)). Task-specific parameters coexist in the same
argparse configuration and are ignored by the other loader.

Fingerprint2Graph derived CSVs contain `molecule_size`,
`molecule_scaffold_size`, and `graph_valid`. Strict loading reads these fields
while constructing its byte-offset index, rather than reparsing every SMILES.
If an older derived CSV lacks them, the loader visibly performs a one-time,
atomic, multiprocessing upgrade before indexing it.

The data package provides a single MassSpecGym store, dataset, collation function,
and Lightning DataModule under `any2graph_v2.data`. During setup, the store
constructs and caches one target graph per unique molecule. Spectrum iteration
therefore reuses the same graph for every spectrum of a molecule instead of
re-running RDKit in every DataLoader worker.

## Quick start

```python
from any2graph_v2.data import build_datamodule
from any2graph_v2.parameter import parse_data_parameters

parameters = parse_data_parameters()
data = build_datamodule(parameters)
data.setup("fit")
batch = next(iter(data.train_dataloader()))

print(batch.tokens.shape, batch.padding_mask.shape, batch.collision_energy.shape)
print(batch.graphs.h.shape, batch.graphs.nodes.labels.shape)
print(batch.graphs.edges.labels.shape)
```

The default training dataset has one item per molecule. Every item access draws
one of that molecule's spectra using NumPy's worker-seeded random generator, so
the association may change between epochs. Validation and test default to one
item per spectrum. Both modes retain global spectrum and molecule indices in
`batch.metadata`.

## Tensor contract

- Annotated spectra: `[B, L, 16]`; raw spectra: `[B, L, 2]`.
- `padding_mask[b,l]` is `True` only for padded spectrum tokens.
- `collision_energy`: float `[B]`, retaining missing CSV values as `NaN` until
  the model maps them to its learned missing-value embedding.
- `graphs.h`: boolean `[B, 48]`, where `True` denotes a real target node.
- `graphs.nodes.labels`: one-hot float `[B, 48, 16]`.
- `graphs.edges.labels`: one-hot float `[B, 48, 48, 5]`, ordered as
  `NO_BOND`, `SINGLE`, `DOUBLE`, `TRIPLE`, `AROMATIC`.
- `graphs.edges.adjacency`: float `[B, 48, 48]` for compatibility and
  inspection; the model's authoritative edge target is the categorical tensor.

Padding is always to the configured graph capacity. Spectrum sequences are
padded to the smaller of the longest sample in the batch and `n_peaks_max`.

## Molecular policies

The atom vocabulary is ordered as `C, N, O, S, P, F, Cl, Br, I, B, Si, As,
Se, Na, K, UNK`. Unsupported atoms map to `UNK` in non-strict mode and cause
the molecule to be removed in strict mode. Unsupported bond types are removed
because the four-class chemical target cannot represent them.

RDKit canonicalizes each SMILES before graph construction. If a target exceeds
`n_nodes_max`, non-strict mode retains the first canonical target atoms and the
induced bonds between them. This can produce a chemically incomplete target and
is recorded as `truncated=True` in sample metadata. Strict mode removes it.

With `scaffold=True`, the target is the Bemis–Murcko scaffold. A ringless
molecule has an empty Murcko scaffold: non-strict mode falls back to the full
molecule, while strict mode removes it. Store-level `policy_counts` makes the
resulting population change visible.

SMILES that RDKit cannot parse are removed by the same policy. RDKit's
process-level diagnostics are suppressed while parsing because they are not
actionable inside a training job; rejection remains visible through
`policy_counts["removed"]`. In particular, isolated explicit hydrogens no
longer produce repeated `not removing hydrogen atom without neighbors`
messages in scheduler logs.

## Command-line arguments

Experiment settings use `argparse`. A script calls `parse_data_parameters()` and
passes the resulting object to the DataModule. Running `--help` shows every
option and its accepted values:

```bash
uv run any2graph-data --help
uv run any2graph-data --batch-size 64 --scaffold --num-workers 4
```

| Argument | Default | Meaning |
|---|---:|---|
| `--data-dir` | `src/data/massspecgym` | dataset directory |
| `--split-method` | `formula` | split CSV basename |
| `--train-iteration-mode` | `molecule` | training unit |
| `--eval-iteration-mode` | `spectrum` | validation/test unit |
| `--spectrum-representation` | `annotated` | `annotated` or `raw` |
| `--n-nodes-max` | `48` | dense graph capacity |
| `--n-peaks-max` | `128` | spectrum token cap |
| `--batch-size` | `32` | DataLoader batch size |
| `--num-workers` | `0` | DataLoader workers |
| `--pin-memory` | disabled | pin host tensors |
| `--scaffold` | disabled | use Murcko targets |
| `--remove-invalid-molecules` | disabled | strict policy |
| `--seed` | `0` | loader and worker seed |

Boolean options also have explicit negative forms such as `--no-scaffold`.
Inside a notebook, construct `DataParameters()` directly because Jupyter itself
adds unrelated command-line arguments.

The molecular vocabularies are stable code-level contracts rather than
experiment options:

```python
from any2graph_v2.parameter import VALID_ATOMS_LIST, VALID_BOND_TYPES
```

The annotated representation is the primary research input because it exposes
subformula information. Raw spectra remain a supported ablation and fallback.
The formula split is the default because it is molecule-disjoint in the local
data and provides a chemically structured generalization test.

## Comparing task sizes at 32 nodes

The `scripts/count_datasets.py` utility prints strict train/validation/test sizes for both
MS2Scaffold and Fingerprint2Mol with `n_nodes_max=32`. From the repository root,
using the project Python environment:

```bash
PYTHONPATH=src python scripts/count_datasets.py
PYTHONPATH=src python scripts/count_datasets.py --fingerprint-file src/data/fingerprint2graph/PUBCHEM_32.csv
```

MS2Scaffold reports both spectra and distinct molecules because spectrum-mode
and molecule-mode training have different epoch sizes. Fingerprint2Mol reports
CSV molecules. Both counts enable `remove_invalid_molecules`, so oversized,
unsupported, malformed, and empty-scaffold targets are excluded rather than
truncated. The default fingerprint source is `4M_32.csv`; use
`--fingerprint-file` to count another preprocessed corpus. An old fingerprint
CSV without persistent graph metadata undergoes the normal one-time parallel
metadata upgrade before it is counted. Use `--massspec-dir`, `--split-method`,
and `--workers` to override the data directory, formula split, and upgrade workers.

## MassSpecGym resource layout

The local loader expects aligned `metadata.csv`, `unique_smiles.csv`,
`spectra.npy`, `annotated_peaks.json`, and `splits/*.csv` resources. Metadata
contains `collision_energy`, `adduct`, `precursor_mz`, `smiles`, and
`unique_smiles_idx`; split files provide a `fold` per spectrum. Preserve row
alignment across spectral resources and split files.

Raw peaks contain `(m/z, intensity)` pairs. Annotated peaks use 16 features:

```text
[m/z / 1000, intensity,
 H / 102, C / 59, O / 25, N / 13, P / 3, S / 6, Cl / 6,
 F / 17, Br / 4, I / 4, B / 1, As / 1, Si / 5, Se / 2]
```

Missing annotations produce one zero token. Missing collision energies remain
`NaN` until the encoder maps them to a learned embedding; present energies use
scale 100 without clipping. Adduct and precursor mass are retained as metadata.

Use a molecule-disjoint split for molecular generalization. The local `random`
split is spectrum-level and can place the same molecule in several folds;
`formula` is the primary split. Ringless molecules have empty Murcko scaffolds:
strict loading filters them, while non-strict loading uses the full molecule as
an explicit fallback. Dataset counts depend on these policies and graph capacity;
use `scripts/count_datasets.py` to inspect the prepared data.

Upstream dataset and loader resources: [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym).
