# Architecture

Any2GraphV2 predicts graphs from Coloring images, molecular fingerprints, and
mass spectra. The prediction branch produces graph logits and node embeddings;
a relational GNN encodes target graphs for learned matching during training.
A detached mirror or Frank–Wolfe solver can supply the alignment instead.

```text
input → modality encoder → graph decoder → graph logits + predicted embeddings
                                                 ↓
target graph → relational GNN → target embeddings → matching plan T[p,t]
                                                 ↓
                                  graph reconstruction objective
```

`T[p,t]` maps predicted slots to target slots. See [loss definitions](LOSS.md)
for objective formulas, [data contracts](DATA.md) for input preparation, and
[configuration](../configs/README.md) for presets and CLI overrides.

## Predictor

The predictor routes Coloring images, molecular fingerprints, or mass spectra to a
modality-specific encoder. Its token memory feeds a shared graph decoder and
heads. Graph capacity `N` comes from the task configuration; targets are only
needed for training alignment and evaluation.

### Quick start

```python
from any2graph_v2.models import build_predictor

model = build_predictor(configuration.data, configuration.model)
prediction = model.forward_batch(batch)
```

The combined data/model command exposes all experiment arguments with defaults
and runs one forward pass:

```bash
uv run any2graph-predict --help
uv run any2graph-predict --batch-size 4 --d-token-input 128 --d-node-decoder 128
```

The model defaults are:

| Argument | Default | Meaning |
|---|---:|---|
| `--graph-decoder-type` | `node_query` | `node_query` or `relationformer` |
| `--d-token-input` | 128 | encoded peak width |
| `--d-node-decoder` | 128 | predicted latent-node width |
| `--n-heads` | 8 | encoder and decoder attention heads |
| `--encoder-layers` | 2 | peak-set encoder layers |
| `--decoder-layers` | 2 | node-query decoder layers |
| `--d-token-input-feedforward` | 256 | input-token Transformer MLP hidden width |
| `--d-node-decoder-feedforward` | 256 | node-decoder Transformer MLP hidden width |
| `--d-edge-decoder` | 64 | bond-head node projection width |
| `--dropout` | 0.1 | Transformer and head dropout |
| `--use-collision-energy` | true | condition peak tokens on collision energy |
| `--collision-energy-max` | 100 | sinusoidal encoder scale; values are not clipped |

Graph capacity remains the data/graph argument `--n-nodes-max`; it is passed to
the model rather than duplicated in `ModelParameters`. Spectrum input width is
derived from the batch: 16 for annotated tokens and 2 for raw peaks. Atom and
bond vocabularies remain fixed constants in `parameter.py`.

### Components

`SpectrumEncoder` applies a small MLP to every peak and then a batch-first
PyTorch Transformer encoder. It deliberately has no sequence-index positional
encoding. The `True = padding` mask is passed to every self-attention layer and
masked output positions are zeroed. By default, the supplied sinusoidal
collision-energy encoder produces one `d_token_input` vector per spectrum. That
vector is added to every projected peak and layer-normalized before
self-attention. Missing energies select a learned vector instead of being
imputed as a physical energy. This sample-level addition preserves peak-order
permutation equivariance.

`NodeQueryDecoder` owns exactly one learned query per graph slot. It projects
encoded peaks to `d_node_decoder` and applies Transformer decoder layers: self-attention
lets slots coordinate, while cross-attention gives every slot access to the
complete spectrum.

`RelationformerDecoder` is the optional Any2Graph-style alternative. It uses
`N+1` DETR-style queries with zero initial states and learned query positions.
The first output is a global relation token; the other `N` outputs retain the
same matcher-facing node-slot contract. Its bond head follows Any2Graph's
undirected construction: each object is concatenated with the relation token
and projected, the two endpoint projections are summed, and a second MLP emits
categorical bond logits. This ports the relation-token mechanism while retaining
the task-specific molecular encoders and losses.

`MolecularGraphHeads` applies independent lightweight heads for presence and
atom labels. A node projection produces `U_i`; the bond head receives symmetric
pair features `U_i + U_j`. Its result is averaged and mirrored from one triangle
to guarantee bit-exact symmetry, rather than relying only on floating-point
symmetry. Diagonal logits are fixed to a strong finite `NO_BOND` state.

There is one authoritative categorical edge prediction with classes
`NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC`. The adjacency logit is derived as
the exact edge-versus-no-edge log odds:

```text
logsumexp(bond_logits[..., 1:]) - bond_logits[..., NO_BOND]
```

This prevents an adjacency head from disagreeing with the bond head.

### Output contract

`SpectrumGraphPrediction` contains:

```text
node_embeddings                   float [B, N, d_node_decoder]
relation_embedding                None, or float [B, d_node_decoder]
graph_logits.h                    float [B, N]
graph_logits.nodes.labels         float [B, N, 16]
graph_logits.edges.labels         float [B, N, N, 5]
graph_logits.edges.adjacency      float [B, N, N]
```

`graph_logits` is a `BatchedDenseData` instance in predicted-slot order. Its
floating `h` values are presence logits, unlike the boolean real-node mask in a
target batch. Names and call sites retain the `graph_logits` distinction.

### Invariances and numerical contract

Because the peak encoder has no index positions, a joint permutation of peak
tokens and their padding mask permutes encoded peak positions but leaves latent
node and graph predictions invariant, up to floating-point reduction error.
Adding arbitrary masked peak
slots likewise does not change real outputs.

Bond and adjacency tensors are bit-exact symmetric. Their diagonal selects
`NO_BOND`. The training objective propagates gradients through latent nodes and graph fields.

## Target graph encoder

The target encoder implements a lightweight relational GNN that maps molecular target
graphs to matcher-width latent node representations. It consumes the existing
`BatchedDenseData` target directly and returns `[B,N,d_node_decoder]`.

### Quick start

```python
from any2graph_v2.models import TargetGraphEncoder
from any2graph_v2.parameter import ModelParameters, TargetEncoderParameters

model_parameters = ModelParameters()
target_encoder = TargetGraphEncoder(
    d_node_decoder=model_parameters.d_node_decoder,
    parameters=TargetEncoderParameters(),
)
target_node_embeddings = target_encoder(batch.graphs)
```

The dual-encoder smoke command exposes data, spectrum-predictor, and target-GNN
arguments through one argparse interface:

```bash
uv run any2graph-encode-targets --help
uv run any2graph-encode-targets --batch-size 4 --target-layers 3
```

The target-specific defaults are:

| Argument | Default | Meaning |
|---|---:|---|
| `--d-node-target` | 128 | message-passing state width |
| `--target-layers` | 3 | relational message-passing layers |
| `--target-dropout` | 0.1 | update dropout |
| `--use-laplacian-pe` | true | append Laplacian symmetry-breaking features |
| `--laplacian-pe-dim` | 8 | retained nonzero Laplacian eigenvectors |

Output width is not duplicated in these parameters. The target encoder receives
the predictor's `d_node_decoder`, guaranteeing the matcher-facing widths agree by
construction.

### Input contract

The encoder requires target semantics:

```text
h                    bool  [B,N]       True = real node
nodes.labels         float [B,N,16]    one-hot atom class
edges.labels         float [B,N,N,5]   one-hot bond/no-bond class
```

Floating presence logits from a predicted graph are rejected. Every graph must
have at least one real node. Padded features are not trusted: real-node and
real-pair masks are reapplied inside the encoder.

### Message passing

Atoms are projected to `d_node_target`. For each layer and each real bond
class `k`, a relation-specific linear projection transforms neighbor `j`. Node
`i` receives:

```text
m_i = (1 / max(degree_i, 1)) * sum_(j,k) bond_ijk * W_k x_j
x_i = LayerNorm(x_i + MLP(concat(x_i, m_i)))
```

`NO_BOND`, self-loops, and every pair involving padding contribute no message.
Degree normalization controls scale across molecular sizes. The residual path
retains atom information for isolated nodes. Padding is reset to exact zero
after initial projection, after every message layer, and after the final
projection to `d_node_decoder`.

### Why dense rather than sparse

`BatchedDenseData` is already required by the matcher and quadratic graph loss,
and `N=48` bounds each graph at only 2,304 ordered pairs. The dense encoder avoids
a second authoritative graph representation, duplicate collation, batch-offset
edge indices, and correspondence tests between sparse and dense node orders.

On the current CPU, the default three-layer encoder processed a real four-graph
batch in roughly 0.01 seconds after data loading. This is only a smoke
measurement, but it provides no current evidence that conversion to a sparse
format is necessary. Training-time profiling may reopen the decision.

### Equivariance and padding

With no positional encoding, every operation is shared across nodes and sums
over neighbors. A simultaneous permutation of `h`, node labels, and both edge
axes therefore permutes the embeddings in the same way. Tests cover permutations
that mix real and padded positions and observe agreement within `1e-6` absolute
tolerance. Increasing only the padded dense capacity leaves real-node outputs
unchanged within the same tolerance.

The encoder supports optional Laplacian positional encoding. It is enabled by
default to provide structural symmetry breaking. The encoder appends
sign-oriented nonzero combinatorial-Laplacian eigenvectors before the atom
projection. Padded modes are shifted above the real graph spectrum, allowing
one padding-aware batched `torch.linalg.eigh` call instead of a Python loop and
one eigensolver launch per graph. The lowest nonzero real modes are selected;
padding and unavailable components remain exactly zero. Raw eigenvectors cannot
define a canonical basis in repeated
eigenspaces, so enabled mode is deliberately treated as symmetry breaking; see
the [matching section](#matching-and-solvers) for the precise rule and limitation. Use
`--no-use-laplacian-pe` for the exact permutation-equivariant ablation described
above.

## Matching and solvers

### Legacy solver comparison

Set `--old-solver-approach` (or `old_solver_approach: true` in YAML) to remove
the target encoder and learned matcher from the model. The training system then
computes

```python
with torch.no_grad():
    T = solver(prediction.graph_logits, target)
```

using the solver selected by `solver_type`. `FrankWolfeSolver` in
`src/solvers/frank_wolfe.py` is adapted from
Any2Graph's PMFGW conditional-gradient loop: it repeatedly linearizes the
quadratic graph cost, solves a linear assignment, and performs an exact segment
line search. Its costs use the current objective's presence,
atom, bond, adjacency, and optional feature-diffusion terms. Its plans have unit
row and column marginals,
so they can be passed directly to the unchanged loss and have zero marginal KL.

The implementation is unified across devices. `solver_backend=cpu` evaluates
the solver in float32 on CPU and uses SciPy's assignment routine;
`solver_backend=gpu` evaluates the same cost construction, Frank–Wolfe updates,
and line searches on CUDA and replaces only the assignment primitive with
`torch_linear_assignment.batch_linear_assignment`. The old path is
nondifferentiable through `T`, matching the original Any2Graph training
semantics, while gradients still flow from the fixed-plan reconstruction loss
into all predictor parameters.

### Mirror solver

`solver_type=mirror` selects `MirrorSolver` in
`src/solvers/mirror.py`. It optimizes the same quadratic objective

\[
F(T)=\langle M,T\rangle+\langle T,K(T)\rangle,
\qquad \nabla F(T)=M+2K(T),
\]

but replaces the Frank--Wolfe assignment and line search by the KL-proximal
mirror step

\[
T_{k+1}=\arg\min_{T\in\mathcal B}
\langle \nabla F(T_k),T\rangle+\tau\,\mathrm{KL}(T\Vert T_k).
\]

For \(\mathrm{KL}(T\Vert T_k)=\sum_{pt}T_{pt}\log(T_{pt}/T_{k,pt})\),
terms independent of \(T\) can be removed to obtain

\[
T_{k+1}=\arg\min_{T\in\mathcal B}
\langle \nabla F(T_k)-\tau\log T_k,T\rangle
+\tau\sum_{pt}T_{pt}\log T_{pt}.
\]

Thus the Sinkhorn kernel is

\[
\exp[-(\nabla F(T_k)-\tau\log T_k)/\tau]
=T_k\odot\exp[-\nabla F(T_k)/\tau].
\]

The last regularizer is negative Shannon entropy. If
\(H(T)=-\sum T\log T\), the rewritten objective contains \(-\tau H(T)\),
not \(+\tau H(T)\). The implementation reuses `sinkhorn_tol`, keeps all soft
solver tensors on CUDA, batches every graph, and freezes each graph when its
maximum plan change is below `tol_outer`. Its validation-only hard projection
is outside the mirror optimization and uses SciPy, like the learned matcher.
Here \(1/\tau\) acts as the mirror step size: large \(\tau\) gives conservative
updates and small \(\tau\) gives sharper, more aggressive updates. Since the
quadratic graph-matching objective is generally nonconvex and this variant has
no line search, a fixed \(\tau\) does not guarantee monotone objective decrease;
it is a scientific hyperparameter to validate empirically.

### Contract

The matcher aligns the 48 predicted slots with the 48 target positions, where
the latter include both real atoms and padding. Its public call uses named
arguments:

```python
result = matcher(
    predicted_node_embeddings=prediction.node_embeddings,
    target_node_embeddings=target_node_embeddings,
    target_padding_mask=~targets.h,
)
```

The cost and transport matrices have orientation `[batch, predicted, target]`:

```text
T[b,p,t] = mass assigning predicted slot p to target position t
```

Consequently, predicted node features align as `T^T F` and predicted edges as
`T^T E T`. A hard permutation stores `permutations[b,t] = p`. This convention
has been checked against both GRALE's reference matcher and this project's
`BatchedDenseData.align_`/`permute_` using a non-self-inverse permutation.

`targets.h` uses `True` for real atoms, whereas `target_padding_mask` uses
`True` for padding. The matcher replaces every padded target embedding with
one learned feature before computing costs. Changing the otherwise meaningless
encoded value at a padding position therefore cannot change a match.

### Sinkhorn matcher

The default follows GRALE's learned matcher:

- learned positional features distinguish predicted slots;
- predicted and target embeddings use separate learned projections;
- projected pairwise L1 distances form the cost;
- when enabled, the Any2Graph feature-diffusion squared-L2 cost is added;
- each cost matrix is divided by its clamped sum;
- log-domain row/column normalizations run for exactly 20 steps by default.

Twenty steps bound the quadratic transport work at `N=48`, but a finite number
of alternating projections does not mathematically guarantee exact row sums.
Every result therefore includes maximum and mean row/column marginal errors.
The objective applies GRALE's `MarginalKL` to penalize these errors
during training. On the real-batch smoke test, both maximum errors were
approximately `2.4e-7`; this observation is not an exactness guarantee.

Cost-sum normalization is protected against the all-zero case. In that case,
Sinkhorn returns the finite uniform matrix rather than NaNs. Soft plans remain
differentiable through the cost, projections, padding embedding, and predicted
positional features.

### Softsort alternative

`--matcher-type softsort` selects the lower-cost GRALE alternative. A learned
scalar score orders target embeddings and a temperature-softened absolute
score difference maps sorted positions to targets. In agreement with the
reference implementation, it does not use predicted content to create the
ordering and normalizes each target column only. Its columns sum to one, while
its row sums need not. Marginal diagnostics and the implemented `MarginalKL`
term are therefore particularly important for this option.

### Hard matching

Evaluation can call `hard_match`, which projects the learned cost to a discrete
one-to-one assignment with SciPy's Hungarian solver. It returns both
target-to-predicted integer indices and equivalent `[B,p,t]` one-hot matrices.
The CPU transfer is intentional: hard matching is nondifferentiable and is for
evaluation metrics, not the training loss.

### Optional Laplacian positional encoding

`TargetGraphEncoder` accepts `--use-laplacian-pe` and
`--laplacian-pe-dim`. The default is enabled. It computes the
combinatorial Laplacian on real target nodes, removes the complete zero
eigenspace, retains the requested lowest nonzero eigenvectors, sign-orients
each vector by its largest-magnitude coordinate, and fills padding or missing
components with zero.

The sign rule resolves an isolated sign flip when the eigenvalue is simple and
the largest-magnitude coordinate is unique. It cannot choose a canonical basis
inside a repeated eigenspace, and ties in the pivot can also depend on node
order. These are intrinsic limitations of raw eigendecomposition for symmetric
graphs. The encoding is therefore an explicit symmetry-breaking feature, not a
claim of exact permutation equivariance. Tests of the core set/graph matching
contract use it disabled; with it disabled, the target GNN and matching plan
are permutation equivariant up to floating-point tolerance.

### Command line

Run the complete data-to-match smoke path with:

```bash
uv run any2graph-match --batch-size 2
uv run any2graph-match --matcher-type softsort --batch-size 2
uv run any2graph-match --use-laplacian-pe --laplacian-pe-dim 8
```

Experiment-varying settings are conventional argparse flags. Stable molecular
vocabularies remain constants in `parameter.py`.
