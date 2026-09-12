# Qualitative checkpoint examples

`any2graph_v2.qualitative_examples_cli` reconstructs the complete experiment
from a self-describing checkpoint and exports fixed validation or test examples.
Every row of the retained overview contains the task input, the hard-aligned
predicted graph, and the ground-truth graph. The exporter also writes each input,
prediction, and target as an independent PNG/PDF file. Figures are accompanied
by JSON provenance and per-example edit-like and exact-reconstruction metrics.
The only plot titles are `Input`, `Prediction`, and `Ground truth`; run names,
metrics, node counts, SMILES, and dataset indices remain in the JSON/W&B
metadata rather than being embedded in the image.

The input panel is task-specific:

- MS2Graph: the mass-spectrum peaks actually retained by `n_peaks_max`;
- Fingerprint2Graph: a 32 x 64 raster of the ECFP4 bits actually passed to the
  fingerprint encoder;
- Coloring: the noisy RGB image reconstructed by the dataset.

Predicted slots are thresholded using the checkpoint's node-presence threshold.
Node and edge classes use argmax. Predictions are then aligned with the same
Hungarian procedure used by validation. This alignment only chooses a display
order; it does not modify predicted node, edge, or presence decisions.
Graph limits are computed from all visible nodes and missing-target markers, so
surplus predicted slots on lower display rows remain fully inside the canvas.
Node-marker area and label size decrease with the number of visible nodes, which
keeps bonds readable in dense molecular and large-capacity Coloring graphs.

Run one checkpoint with:

```bash
.venv/bin/python -m any2graph_v2.qualitative_examples_cli \
  --checkpoint artifacts/<RUN>/checkpoints/best-<EPOCH>-<STEP>.ckpt \
  --selection distinct_targets \
  --num-examples 4 \
  --device cuda \
  --output-dir paper_figures/<RUN> \
  --wandb-mode online \
  --wandb-project any2graph-v2 \
  --wandb-run-name <RUN>-qualitative
```

`--data-dir` and `--data-file` are runtime path overrides when the paths stored
inside the checkpoint differ on the evaluation machine. Fixed indices should be
reported with the checkpoint and should not be selected after viewing model
predictions.

`--selection distinct_targets` scans the chosen split from index zero and keeps
the first requested target identities. MS2Graph uses `molecule_index`, so
multiple collision-energy spectra of one molecule cannot occupy several rows.
Fingerprint2Graph uses canonical target SMILES. Coloring uses the exact stored
labeled graph. The resolved dataset indices and identities are written to JSON
and the W&B metric table, making automatic selection reproducible. Use
`--selection indices --indices ...` only when an explicitly reported fixed set
is required.

The repository also provides one SLURM job for the selected MS2Graph,
PubChem32, and N20 Coloring models:

```bash
sbatch scripts/export_selected_qualitative_examples.sh
```

The job resolves the unique `best-*.ckpt` in each run directory, where “best”
means minimum epoch-aggregated validation loss under training's checkpoint
callback. Results are written under `paper_figures/selected_best_models/`.
It creates one W&B run per source checkpoint in the
`selected-best-model-qualitative` group. Each W&B run logs the combined PNG
panel under `qualitative/panels`, a row-per-example image table under
`qualitative/example_triplets`, a per-example metric table, and a versioned
downloadable artifact containing every PNG/PDF plus JSON provenance. Within the
artifact, each input and its two corresponding graphs share a directory such as
`examples/example-00-index-0/{input,prediction,ground-truth}.png`; PDF versions
use the same directory. Overview files live under `overview/` and provenance
under `metadata/`. Local separate files retain names such as
`<RUN>-example-00-index-0-input.png`, `...-prediction.png`, and
`...-ground-truth.png`. Set
`WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`, or `WANDB_MODE` in the `sbatch`
environment to override their defaults.
