# Fingerprint2Graph task

## Purpose

This task is a controlled simplification of the Any2GraphV2 research task.
The MassSpecGym scaffold experiment plateaued near edit distance 24, so this
branch tests whether the graph decoder, learned GRALE matcher, and corrected
permutation-aware loss can learn when the input is a deterministic molecular
fingerprint rather than a noisy mass spectrum.

Mass spectra and acquisition metadata are not model inputs for this task. The
target graph, target encoder, matcher, reconstruction objective, Lightning
training, metrics, checkpointing, and W&B integration remain unchanged.

## Input contract

The implementation follows
the upstream [Any2Graph repository](https://github.com/KrzakalaPaul/Any2Graph) (`Any2Graph/Fingerprint2Graph`):

1. parse the target SMILES with RDKit;
2. compute an ECFP4-style hashed Morgan bit fingerprint with radius 2 and 2,048
   bins;
3. extract the sorted active bit indices;
4. prepend SOS token 2,049 (`UNK=2,048`, vocabulary size 2,050);
5. embed token IDs at width `d_token_input`;
6. add fixed sinusoidal sequence positions and apply the Transformer encoder;
7. pass the encoded tokens to the unchanged learned node-query decoder.

Padding uses integer zero. Morgan bit zero is also a valid token, as in the
reference. They remain unambiguous because `padding_mask`, not token value,
defines padding. Sequences longer than `--n-tokens-max` retain SOS and the first
sorted active bits; truncation is recorded in batch metadata.

Every derived CSV includes reusable graph metadata:

- `molecule_size`: canonical full-molecule heavy-atom count;
- `molecule_scaffold_size`: Murcko-scaffold heavy-atom count, with `0` for a
  ringless molecule and `-1` only if scaffold construction failed;
- `graph_valid`: whether the complete molecule uses the strict atom and bond
  vocabulary.

Strict loading uses these fields to build its byte-offset index without RDKit:
full-molecule mode retains `graph_valid` rows with
`0 < molecule_size <= n_nodes_max`; scaffold mode instead uses
`molecule_scaffold_size`. This prevents ringless strict-scaffold rows from
failing later in DataLoader workers.

Older derived CSVs are upgraded once, in place, when strict loading first needs
them. The upgrade streams the input through a bounded process pool, displays a
`tqdm` counter, writes a temporary file, and atomically replaces the old CSV
only after success. The raw source is never modified. Without strict removal,
the established behavior is retained: ringless molecules fall back to their
complete molecular graph.

## Raw-data preprocessing

Two raw source formats are supported:

- `src/data/fingerprint2graph/4M_raw.csv`: named CSV containing at least
  `smiles`;
- `src/data/fingerprint2graph/PUBCHEM_raw.csv`: tab-separated PubChem ID and
  SMILES despite its `.csv` suffix. This is downloaded automatically from
  PubChem when the preprocessor is run without an input path.

Preprocessing parses rows as a stream. A bounded process pool parallelizes the
expensive RDKit parsing, molecular metadata calculation, and heavy-atom
counting, while the parent process keeps input ordering, fold assignment, and
CSV writing deterministic. It removes
invalid SMILES and keeps only molecules with strictly fewer than 32 heavy
atoms. Validation mirrors the graph builder: after removing explicit
hydrogens and canonicalizing, molecules with no heavy atoms and molecules
containing bond types outside `SINGLE`, `DOUBLE`, `TRIPLE`, and `AROMATIC` are
also removed. This catches syntactically valid but unusable inputs such as the
hydrogen-isotope-only PubChem CID 159980138. Atoms outside the fixed vocabulary
remain supported through the model's `UNK` atom class, matching non-strict
training configurations. The preprocessor preserves source columns and adds
`fold`. Each accepted row is independently assigned with seed 0 to
train/validation/test with probabilities 90%/5%/5%. Thus the split is
deterministic for a fixed row order but counts are probabilistic rather than
forced to exact ratios.

```bash
source .venv/bin/activate
python -m any2graph_v2.preprocess_fingerprints \
  src/data/fingerprint2graph/4M_raw.csv --atom-limit 32 --seed 0 --workers 8

# Download the official CID--SMILES archive if needed, then run the larger set:
python -m any2graph_v2.preprocess_fingerprints \
  --atom-limit 32 --seed 0 --workers 8
```

With no explicit output, these write `4M_32.csv` and `PUBCHEM_32.csv` next to
their inputs. Output is streamed and raw files are never overwritten.
`--workers` defaults to the detected logical CPU count, `--workers 1` provides
the serial debugging path, and `--worker-batch-size` controls IPC granularity
(default 2,000 rows). At most two batches per worker are queued, so increasing
the dataset size does not create an unbounded future queue.

A `tqdm` display is enabled by default and reports percentage, ETA, processed
rows, throughput, and live kept/oversized/invalid counts. Here `invalid`
aggregates parse failures, empty molecules, unsupported bond types, and caught
per-record RDKit exceptions. In particular, a failure during hydrogen removal,
canonicalization, or bond inspection discards only the affected SMILES; it does
not terminate the worker or the preprocessing run. The final summary reports
these exceptional rows separately as `rdkit_error`. The script first performs
a fast line-count scan because both supplied formats contain exactly
one record per physical line. Use `--no-count-total` to skip that scan and show
only count/rate, `--no-progress` for quiet jobs, and `--progress-every` to
control how often the filter-count postfix is refreshed.

The executed seed-0 preprocessing of `4M_raw.csv` produced:

| Outcome | Rows |
| --- | ---: |
| source | 4,172,787 |
| kept | 2,770,628 |
| train | 2,493,824 |
| validation | 138,365 |
| test | 138,439 |
| removed at 32+ heavy atoms | 1,402,158 |
| invalid SMILES | 1 |

The no-argument command first checks for
`src/data/fingerprint2graph/PUBCHEM_raw.csv`. If it is absent, it downloads
PubChem's official `CID-SMILES.gz` archive over HTTPS, streams the archive to a
temporary file, unpacks it, and atomically publishes the raw TSV. It prints
whether the file was reused, downloaded, unpacked, and made ready. The archive
and unfinished temporary files are not retained. This is a large download and
the unpacked raw file is approximately 8.1 GB.

Older preprocessed files are upgraded automatically with the three metadata
columns when strict loading needs them. Regenerate from raw data only when the
current preprocessing filter itself must be applied retroactively (for example,
to remove rows admitted before unsupported-bond filtering was introduced):

```bash
python -m any2graph_v2.preprocess_fingerprints \
  src/data/fingerprint2graph/PUBCHEM_raw.csv --atom-limit 16 --seed 0 --workers 8
```

## Training

The supplied `default.yaml` currently uses 16 graph slots and `PUBCHEM_16.csv`.
The 32-slot examples below remain useful for the 4M dataset. Because
preprocessing enforces an atom limit, every retained molecule fits the matching
capacity when the same limit is used for training.

```bash
python -m any2graph_v2.data_cli --data-file src/data/fingerprint2graph/4M_32.csv
python -m any2graph_v2.predict_cli --data-file src/data/fingerprint2graph/4M_32.csv
python -m any2graph_v2.train_cli --config default
```

`--no-use-token-positions` is the principal input-encoder ablation. Set
`--task fingerprint2graph` to select this dataloader and encoder. Setting
the same configuration field to `ms2graph` selects MassSpecGym and the spectrum
encoder; all downstream graph prediction and training components are shared.

Run the paper's 32-slot configurations with:

```bash
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-fingerprint2mol-matcher.yaml
uv run --frozen python -m any2graph_v2.train_cli \
  --config experiments/paper/configs/main-fingerprint2mol-mirror.yaml
```

See the [experiment guide](../experiments/paper/README.md) for budgets and overrides.
