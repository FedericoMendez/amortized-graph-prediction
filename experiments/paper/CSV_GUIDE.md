# Paper CSVs

The CSV files collect the results and experimental settings reported in
*Graph Matching Relaxations and Amortization for Supervised Graph Prediction*.
File names refer to the corresponding tables and figures in the paper.

Each CSV includes source page/table references. GI accuracy uses percentage
units; `0.001` means 0.001%, not 0.1%. Edit distance is reported as printed.
Blank values encode missing information. In Table 1, blanks correspond to
FGWBARY task combinations marked with a dash in the paper. Printed numeric
precision may be normalized (for example, `02.64` becomes `2.64`).

The settings tables also generate [runnable experiment configurations](README.md).
The generator excludes metric columns and documents the remaining implementation choices.

## Contents

- Tables 1 and 2: reported scores in tidy task/method or task/loss rows.
  Table 2 retains the paper's argument order. The runnable configs use the
  explicit implementation mapping documented in the experiment guide.
- Table 3: dataset counts from the draft.
  The fingerprint count of 80,897,368 is transcribed exactly; verify the
  associated corpus version, filtering, and split cardinalities before use.
- Table 4: shared architecture/training settings. Mixed text/numeric columns
  are intentional. Table 5 gives capacity-dependent Coloring budgets.
- Table 6: the Cartesian grid of six mirror and eight matcher configurations.
  The mirror 100-inner/100-outer row is retained with `included_in_plot=false`,
  following the draft. Latency and edit-distance cells are blank: underlying
  per-run values were not supplied. These rows describe the configuration grid.
- Table 7: rounded epsilon candidates in units of 1e-6, exactly as printed.
  Multiply by 1e-6 when configuring code. Do not treat rounded candidates as
  the actual selected epsilon or regenerate them with an unreported rounding rule.
- Figure 2 right template: six capacities by two methods, with documented solver
  settings. Selected matcher epsilon, scores, and checkpoints remain blank.
- Figure 3: qualitative input/prediction/target examples are images, not a numeric
  experiment table. Associate example IDs
  and checkpoint hashes with the existing qualitative exporter when the source
  run is identified. Figure 1 is an architecture diagram, not an experiment.

## Configuration notes

1. The draft says the ground discrepancy is cross-entropy (page 5). Current
   relaxed objectives use probability-space KL terms, whose target entropy can
   depend on the soft plan. Equivalence must not be assumed for soft targets.
2. Table 4 reports 10 Fingerprint2Mol epochs; the relaxed/prime molecular launchers
   currently use 8. The paper additionally states half budgets for Any2Graph.
3. Table 4 reports minimum learning rate 1e-5; the default preset is 0.0.
   Resolved run settings, rather than the preset alone, determine this value.
4. The paper's objective weights include adjacency 0.5. The default YAML sets
   0.5, but bare `ObjectiveParameters()` uses 0.2. The smoke workflow uses
   code defaults for its small CPU check.
5. Main-text target-GNN wording says three layers, while Table 4 specifies five
   for Coloring. Resolve using the actual runs and update the manuscript.

The manifest tracks the run records needed to resolve these configuration details.
