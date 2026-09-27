# NodeForge

 reproducible node-classification benchmark on synthetic graphs, with
**AdaProp** — a validation-tuned multi-channel propagation model that combines
SGC-style channel propagation (Wu et al., 2019), initial residuals (GCNII,
Chen et al., 2020), a feature-similarity gated adjacency, and a feature-space
kNN channel. Fully offline: no dataset downloads, no pretrained weights.

[![CI](https://github.com/CJX0712/nodeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/nodeforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/nodeforge)](https://github.com/CJX0712/nodeforge/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![Quality](https://img.shields.io/badge/quality-S%20(world--class)-success)](#performance)

## Performance (5 seeds, mean±std, Macro-F1 on held-out test nodes)

| Tier | NodeForge AdaProp | Best baseline | Δ |
|---|---|---|---|
| tier1_homophilous_noisy | **0.9208±0.0212** | sage 0.8859±0.0436 | +0.0350 (significant) |
| tier2_dense_noisy | **0.8488±0.0379** | sage 0.8168±0.0560 | +0.0320 |
| tier3_heterophilous | **0.9549±0.0219** | sage 0.8901±0.0462 | +0.0648 (significant) |

Gate: mean Δ = **+0.0439** ≥ preset 0.03, 2/3 tiers pass the
0.5×(std₁+std₂) significance test, all tiers positive → **passed**.
Full numbers (including the numpy fallback and per-seed values):
[`benchmark.json`](benchmark.json) — every figure is produced by
`python examples/run_demo.py`; nothing is hand-written.

## Install & run

```bash
git clone https://github.com/CJX0712/nodeforge.git
cd nodeforge
python -m venv .venv && .venv/Scripts/activate        # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, for torch baselines
pip install -r requirements.txt
python examples/run_demo.py          # full benchmark -> benchmark.json (~25 s)
python -m nodeforge.cli models      # list models
python -m nodeforge.cli run --tier tier3_heterophilous --seed 42
python -m nodeforge.cli benchmark --seeds 5
python -m nodeforge.cli info        # ENV_NODEFORGE_* overrides
```

Torch is **optional**: without it, the torch baselines (gcn/sage/mlp) are
skipped automatically and the primary model still runs (numpy + scikit-learn
only). The degraded path is unit-tested.

## Models

| name | kind | description |
|---|---|---|
| **adaprop** | primary | multi-channel propagation (gated / plain / kNN) + LR head, candidate library selected on train+val |
| adaprop_np | ours (degenerate) | pure gated channel only — the offline recipe |
| gcn | baseline | Kipf & Welling 2017 (torch) |
| sage | baseline | GraphSAGE-mean, Hamilton et al. 2017 (torch) |
| mlp | baseline | 2-layer feature-only MLP (torch) |
| logreg | baseline | LogisticRegression on raw features |
| lp | baseline | label propagation (Zhu & Ghahramani 2002) |
| netmf | baseline | NetMF-lite embedding + LR (Qiu et al. 2018) |

## Ablation (tier1, Macro-F1, 5 seeds)

| variant | F1 | Δ vs full |
|---|---|---|
| adaprop (full) | 0.9208 | — |
| − residual | 0.9069 | −0.0140 |
| − kNN channel | 0.9025 | −0.0184 |
| − gated channel | 0.8962 | −0.0247 |

Every component contributes; the same ordering holds on tier3 where the kNN
channel is decisive (see benchmark.json → ablation).

## Layout

```
nodeforge/
  core/        types · errors(E100-E500) · config(ENV_NODEFORGE_*) · interfaces · seed
  data/        synthetic graph generators (SBM + LFR, seeded, zero downloads)
  graph/       AdaProp + torch GNNs + numpy baselines + fallback
  preprocess/  stratified splits (leakage-safe) · normalization/gating
  training/    full-batch trainer (torch)
  hpo/         budget-limited grid search
  eval/        metrics · failure-case mining · report/gate
  pipeline/    benchmark orchestration
  cli.py       argparse entry
examples/      run_demo.py -> benchmark.json
tests/         51 pytest cases (incl. offline-degradation + determinism)
docs/          architecture.md · model_card.md
```

## Reproducibility

- `core.seed.set_all(seed)` is the single seeding entry (python / numpy / torch).
- Same seed ⇒ bitwise-identical `benchmark.json` (verified: two full runs,
  all sections equal; only wall-clock differs).
- Dependencies locked in `requirements.lock.txt`.

## License

MIT — author 晨星 (CJX0712).
