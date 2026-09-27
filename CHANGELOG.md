# Changelog

## 0.1.0 — 2026-09-28

Initial release.

- AdaProp: multi-channel propagation (gated / plain / kNN) with candidate
  library selected on train+val, LR head; backend-free (numpy+sklearn).
- Baselines: GCN / GraphSAGE / MLP (torch, optional), LogReg, Label
  Propagation, NetMF-lite; pure-gated numpy fallback `adaprop_np`.
- Seeded synthetic benchmark: SBM + LFR generators, 3 difficulty tiers
  (homophilous-noisy / dense-noisy / heterophilous).
- Gate: mean Macro-F1 Δ ≥ +0.03 vs best external baseline with std-based
  significance on ≥2/3 tiers and all-positive deltas — passed on 5 seeds
  (mean Δ = +0.0439).
- Ablation suite (nogate / noknn / nores) + failure-case mining with cause
  attribution.
- Deterministic end-to-end: same seed reproduces benchmark.json bitwise
  (except wall-clock). 51 tests, ruff clean, coverage 93%.
