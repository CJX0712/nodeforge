# Model Card — NodeForge / AdaProp

## Model

- **Name**: AdaProp (validation-tuned multi-channel propagation).
- **Type**: transductive semi-supervised node classifier. Linear head over
  SGC-style propagation channels built from a mixed adjacency (gated existing
  edges, plain adjacency, feature-space kNN graph) with initial residuals.
- **Innovation being evaluated**: the candidate-library channel mix
  (gated / plain / kNN / residual weights) selected on train+val macro-F1.
  Component value is established by ablation; no novel-architecture claim.
- **Built from published techniques**: SGC (Wu et al. 2019), GCNII initial
  residuals (Chen et al. 2020), label propagation (Zhu & Ghahramani 2002),
  NetMF (Qiu et al. 2018), GCN (Kipf & Welling 2017), GraphSAGE (Hamilton et
  al. 2017), kNN feature graphs from the heterophily literature.

## Intended use

- Research and engineering baseline for node classification on graphs whose
  features carry class signal (including heterophilous graphs).
- Offline-safe: all data are generated locally; the primary model needs only
  numpy + scikit-learn. Torch is used for comparison baselines.

## Data

- Synthetic, seeded generators (SBM, LFR). No personal data, no downloads.
- Three fixed tiers (see architecture.md); tier difficulty was calibrated on
  development seeds 42–44 before the final 5-seed evaluation (42–46).

## Metrics (final run, 5 seeds, Macro-F1 on test nodes)

| tier | AdaProp | best external baseline | Δ | significant |
|---|---|---|---|---|
| tier1_homophilous_noisy | 0.9208±0.0212 | sage 0.8859±0.0436 | +0.0350 | yes |
| tier2_dense_noisy | 0.8488±0.0379 | sage 0.8168±0.0560 | +0.0320 | no |
| tier3_heterophilous | 0.9549±0.0219 | sage 0.8901±0.0462 | +0.0648 | yes |

Gate (pre-registered): mean Δ ≥ 0.03 ∧ ≥2 significant tiers ∧ all positive →
**passed** (mean Δ = 0.0439).

## Limitations

- Synthetic benchmarks with Gaussian prototype features: conclusions may not
  transfer to real-world graphs with text/citation features.
- Transductive only: new nodes require a refit (no inductive inference).
- Dense adjacency: memory O(n²) — practical to n ≈ 5–10k nodes on CPU.
- Class-balance sensitivity: `analyze_failures` flags minority-class errors,
  but the head is not cost-sensitive.
- kNN channel adds O(n²) cosine computation per fit.

## Ethical considerations

None specific: no human data, no pretrained weights, no external services.
