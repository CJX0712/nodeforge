# NodeForge Architecture

## Dependency graph (one-way, acyclic)

```
cli ──► pipeline ──► {data, preprocess, graph, training, hpo, eval} ──► core
```

No module imports upward. `core` imports nothing from the package.

## core/

| module | contents |
|---|---|
| `types.py` | `GraphBundle` (x, edge_index, y; validation in `__post_init__`), `SplitMasks` (disjointness + coverage enforced) |
| `errors.py` | `NodeForgeError` base; E100 data / E200 model / E300 training / E400 eval / E500 pipeline-config |
| `config.py` | `NodeForgeConfig` dataclass; every field overridable via `ENV_NODEFORGE_*`; schema validation raises E500; pre-registered gate constants `F1_GATE=0.03`, `SIGNIFICANT_FRAC=2` |
| `interfaces.py` | `NodeClassifier` protocol: `fit(bundle, masks)` / `predict(bundle) -> int64 labels` |
| `seed.py` | `set_all(seed)` — python + numpy + torch seeding, returns a `np.random.Generator`; also caps torch threads for tiny full-batch graphs |

## data/ — generators (all seeded, zero downloads)

- `make_sbm(n_per_class, p_in, p_out, feat_noise, seed)` — 6-block SBM, classes = blocks, features = class prototypes + Gaussian noise. `p_in < p_out` produces heterophilous graphs.
- `make_lfr(n, mu, ...)` — LFR benchmark (Fortunato et al.) with a deterministic retry chain (seed*100+i) and E100 after 50 failures.
- `get_benchmark_tiers()` — the three gated tiers (fixed before the final multi-seed evaluation):

| tier | params | intent |
|---|---|---|
| tier1_homophilous_noisy | p_in=0.045, p_out=0.010, noise=2.0 | homophily + heavy feature noise |
| tier2_dense_noisy | p_in=0.060, p_out=0.020, noise=2.2 | denser in-class edges, more noise |
| tier3_heterophilous | p_in=0.002, p_out=0.070, noise=1.8 | anti-homophilous edges, informative features |

## preprocess/

- `make_masks` — stratified 20/class train, seeded val, rest test; masks drawn once per seed, never refit.
- `symnorm` — D^-1/2 (A+I) D^-1/2.
- `gated_feature_weights` — W_ij = exp(cos(x_i,x_j)/tau) on existing edges.

## graph/ — models

- `AdaPropClassifier` (primary, backend-free numpy+sklearn):
  1. channel families: **gated** (exp(cos/tau) re-weighted edges), **plain** (symnorm A), **knn** (symmetrized cosine-kNN graph in raw feature space);
  2. for each active candidate (gated, plain, knn, residual) ∈ CANDIDATES (9 rows, weights sum to 1): P = g·A_gated + p·A_plain + k·A_knn; channels [x, relu(Px), relu(P²x), relu(P³x)] with initial residual h ← (1−r)·relu(Ph) + r·x (GCNII-style, SGC-style linear propagation);
  3. channels standardized with train-only stats; convex LogisticRegression head (C=1);
  4. candidate with best train+val macro-F1 wins (test untouched).
- Torch baselines: `GCNClassifier`, `SAGEClassifier` (mean aggregation), `MLPClassifier` — full-batch, early stopping on val accuracy, train-only feature scaling.
- Numpy baselines: `LogRegBaseline`, `LabelPropagation`, `NetMFLR` (SVD of (P+P²+P³)/3).
- `NumpyPropClassifier` (`adaprop_np`) — pure gated channel, fixed residual: the degenerate single-channel recipe and offline reference.

## training/ · hpo/ · eval/

- `train_full_batch` — Adam, CE on train rows, early stop on val accuracy, NaN → E300; optional per-group param groups.
- `grid_search` — deterministic candidate order, budget-limited, E500 on bad budget.
- `metrics` — accuracy / macro-F1 (sklearn imported under aliases).
- `failure_cases.analyze_failures` — up to 3 misclassified test nodes per tier with automatic attribution: low degree (<P25) → minority class (only when class frequencies actually differ) → heterophilous neighborhood (<0.5 same-class neighbor share) → generic.
- `report.judge` — gate: mean Δ over tiers ≥ `F1_GATE`, ≥ `SIGNIFICANT_FRAC` tiers with mean diff > 0.5×(std₁+std₂), all tier deltas positive. Models named `adaprop*` are excluded from the baseline set (they are the system's own).

## pipeline/

`NodeForgePipeline.benchmark()` — for tier × seed: `set_all(seed)` → generate graph → masks → fit/predict every available model (torch models auto-skipped when torch is missing) → per-seed metrics; then ablation variants, failure cases, gate verdict, environment metadata. Output is JSON with sorted keys; two runs are bitwise-identical in every section except `timing_sec`.

## Determinism notes

- LFR retries and all RNG flow from the single seed; no wall-clock, hash-order or thread-count dependence in results (torch threads only affect speed).
- `scipy.sparse.linalg.svds` (ARPACK) is deterministic for a fixed matrix.
