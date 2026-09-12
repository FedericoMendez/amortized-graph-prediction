# Permutation-aware reconstruction loss

## Authoritative contract

`GraphReconstructionObjective` is a molecular-field port of the corrected
GRALE objective in the upstream [GRALE loss implementation](https://github.com/KrzakalaPaul/GRALE) (`GRALE/loss`). It is selected with
`--reconstruction-loss original`. The legacy
`GRALE_old` training path, which first mixed output logits and then evaluated an
aligned loss, is deliberately not reproduced.

The matcher plan uses `T[b,p,t]`: rows are predicted slots and columns are
target positions. Every loss primitive consumes `T` directly:

```python
result = objective(
    predicted_graph=prediction.graph_logits,
    targets=batch.graphs,
    plan=matching.plan,
)
result.loss.backward()
```

The plan provider is interchangeable. By default it is the learned matcher; in
the `old_solver_approach` comparison it is the configured no-gradient solver,
with either a CPU or GPU assignment backend. Both return the same unit-marginal
`T[p,t]` contract, so this objective and all normalizations remain unchanged.

## Alternative relaxed objectives

For predicted edge probabilities `A`, target edge probabilities `B`, and
transport `T`, the four alternatives implement

```text
alt_a:       KL(T @ B || A @ T)
alt_b:       KL(T @ B @ T.T || A)
alt_a_prime: KL(B @ T.T || T.T @ A)
alt_b_prime: KL(B || T.T @ A @ T)
```

`alt_a` and `alt_b` retain the direct-plan presence, node-label, and optional
feature-diffusion components. Following their reverse-direction definitions,
`alt_a_prime` transports target node fields into predicted order, whereas
`alt_b_prime` transports predicted node fields into target order. All variants
retain the same raw-plan marginal KL component and result/logging schema.

The same construction is applied to categorical edge labels and binary
adjacency. Computation is in probability space; logits are converted with
softmax or sigmoid before alignment. Target padding is removed before
transport, and transported values are conditioned on their real-node mass.
Because bounded Sinkhorn plans can retain small marginal errors, the edge
alignment uses row- or column-normalized copies of `T`; the marginal KL still
uses the raw plan and therefore continues to penalize those errors.

The alternative losses require learned matching. The detached Frank--Wolfe
and mirror solvers optimize the original quadratic objective internally, so
combining either solver with any alternative is rejected instead of silently
using a mismatched alignment.

## Components

For target presence `h_t`, predicted presence logits `a_p`, atom logits `F_p`,
target one-hot atoms `Y_t`, predicted bond logits `E_pq`, target one-hot bonds
`B_tu`, predicted adjacency logits `A_pq`, and binary target adjacency `G_tu`:

```text
L_presence  = sum_pt   T_pt BCE(a_p, h_t) / N
L_atom      = sum_pt   T_pt h_t CE(F_p, Y_t) / n
L_bond      = sum_pqtu T_pt T_qu h_t h_u CE(E_pq, B_tu) / n^2
L_adjacency = sum_pqtu T_pt T_qu h_t h_u BCE(A_pq, G_tu) / n^2
L_FD        = sum_pt   T_pt h_t ||D_p - (GF)_t||_2^2 / n  (optional)
L_marginal  = MarginalKL(T)

L = alpha_presence  * L_presence
  + alpha_atom      * L_atom
  + alpha_bond      * L_bond
  + alpha_adjacency * L_adjacency
  + alpha_FD        * L_FD
  + alpha_marginal  * L_marginal
```

Here `N` is the padded capacity and `n=sum_t h_t`. Target presence masks padded
atom labels and both axes of the edge objectives. The categorical bond tensor
contains `NO_BOND`; adjacency logits are derived exactly from that same tensor,
so the separate GRALE adjacency term adds no independent prediction head. It
does deliberately give the bond-versus-no-bond decision its own weight.

Feature diffusion follows Any2Graph: `F` is the target node-label matrix, `G`
is the raw target adjacency, and an independent predictor head produces `D`.
The target `G @ F` is computed online, so enabling the term does not require
regenerating datasets. The same squared-L2 pairwise cost is supplied to the
selected matcher.

GRALE includes diagonal pairs by default. `--exclude-self-loops` remains
available as an explicit molecular ablation and masks both predicted and target
diagonals while retaining the reference `n^2` denominator.

## Reference implementation reuse

The local `LinearBCE`, `LinearCE`, `QuadraticBCE`, `QuadraticCE`, and
`MarginalKL` preserve the corrected reference formulas. Quadratic losses use
GRALE's factorization

```text
L(a,b) = f1(a) + f2(b) - <h1(a), h2(b)>
```

and its tensor-product contraction, avoiding a materialized six-dimensional
pairwise edge-loss tensor. Automated oracle tests evaluate both self-loop modes
and compare every primitive numerically with the classes imported from
the upstream [GRALE loss implementation](https://github.com/KrzakalaPaul/GRALE) (`GRALE/loss`).

The corrected branch has no column-entropy term. `MarginalKL` is:

```text
sum_rows    (-log(m) + m - 1)
+ sum_cols (-log(m) + m - 1)
```

and is evaluated directly on the matcher plan. As in GRALE, a zero marginal
has infinite loss; the configured matchers must produce strictly positive
marginals.

## Defaults and CLI

| Argument | Default | Meaning |
|---|---:|---|
| `--reconstruction-loss` | `original` | direct objective or one of `alt_a`, `alt_b`, `alt_a_prime`, and `alt_b_prime` |
| `--alpha-presence` | 1.0 | node-presence BCE |
| `--alpha-atom` | 1.0 | atom-label CE |
| `--alpha-bond` | 0.2 | categorical edge-label CE |
| `--alpha-adjacency` | 0.2 | binary adjacency BCE |
| `--alpha-marginal` | 1.0 | row/column marginal KL |
| `--feature-diffusion` | false | enable Any2Graph-style `A @ F` supervision |
| `--alpha-feature-diffusion` | 1.0 | squared-L2 diffusion weight |
| `--exclude-self-loops` | false | optional edge-diagonal mask |

All weights must be nonnegative and at least one must be positive. Raw
components, per-graph values, target sizes, and marginal errors are retained in
the result. Run an end-to-end real-batch check with:

```bash
uv run any2graph-loss --batch-size 2
uv run any2graph-loss --feature-diffusion --batch-size 2
uv run any2graph-loss --matcher-type softsort --batch-size 2
uv run any2graph-loss --reconstruction-loss alt_a --batch-size 2
uv run any2graph-loss --reconstruction-loss alt_b --batch-size 2
uv run any2graph-loss --reconstruction-loss alt_a_prime --batch-size 2
uv run any2graph-loss --reconstruction-loss alt_b_prime --batch-size 2
```
