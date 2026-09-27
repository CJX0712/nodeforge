"""Aggregation and report rendering. All stats come from real run outputs."""
from __future__ import annotations

from typing import Any

import numpy as np

from nodeforge.core.config import F1_GATE, SIGNIFICANT_FRAC


def aggregate(per_seed: list[dict]) -> dict:
    """per_seed: list of {"acc": float, "macro_f1": float} -> mean/std blocks."""
    out: dict[str, Any] = {"n_seeds": len(per_seed)}
    for key in ("acc", "macro_f1"):
        vals = np.array([r[key] for r in per_seed], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "per_seed": [float(v) for v in vals],
        }
    return out


def significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """Simple pre-registered gate: mean diff > 0.5 * (std_a + std_b)."""
    return (mean_a - mean_b) > 0.5 * (std_a + std_b)


def judge(results: dict) -> dict:
    """Primary gate: AdaProp vs best EXTERNAL baseline per tier (models of the
    adaprop family are the system's own and are excluded from the baseline
    set), then the global verdict."""
    tiers = sorted(results.keys())
    gate = {"per_tier": {}, "threshold": F1_GATE, "significant_tiers": 0}
    deltas = []
    for tier in tiers:
        models = results[tier]
        if "adaprop" not in models:
            continue
        a = models["adaprop"]["macro_f1"]
        baselines = {m: v["macro_f1"] for m, v in models.items()
                     if not m.startswith("adaprop")}
        best_b = max(baselines, key=lambda m: baselines[m]["mean"])
        b = baselines[best_b]
        sig = significant(a["mean"], a["std"], b["mean"], b["std"])
        gate["per_tier"][tier] = {
            "adaprop_mean": a["mean"], "best_baseline": best_b,
            "baseline_mean": b["mean"], "delta": a["mean"] - b["mean"],
            "significant": bool(sig),
        }
        deltas.append(a["mean"] - b["mean"])
        gate["significant_tiers"] += int(sig)
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    gate["mean_delta"] = mean_delta
    gate["all_positive"] = bool(deltas) and all(d > 0 for d in deltas)
    gate["passed"] = bool(
        mean_delta >= F1_GATE and gate["significant_tiers"] >= SIGNIFICANT_FRAC
        and gate["all_positive"]
    )
    return gate


def markdown_table(results: dict) -> str:
    """Render the mean±std matrix as markdown."""
    tiers = sorted(results.keys())
    lines = ["| tier | model | acc | macro_f1 |", "|---|---|---|---|"]
    for tier in tiers:
        for model in sorted(results[tier]):
            r = results[tier][model]
            a, f = r["acc"], r["macro_f1"]
            lines.append(
                f"| {tier} | {model} | {a['mean']:.4f}±{a['std']:.4f} "
                f"| {f['mean']:.4f}±{f['std']:.4f} |"
            )
    return "\n".join(lines)


def cli_table(results: dict) -> str:
    """Fixed-width aligned table for the terminal (pitfall: header alignment)."""
    tiers = sorted(results.keys())
    header = f"{'tier':<24}{'model':<12}{'acc':<16}{'macro_f1':<16}"
    lines = [header, "-" * len(header)]
    for tier in tiers:
        for model in sorted(results[tier]):
            r = results[tier][model]
            a, f = r["acc"], r["macro_f1"]
            lines.append(f"{tier:<24}{model:<12}"
                         f"{a['mean']:.4f}±{a['std']:.4f}{'':<6}"
                         f"{f['mean']:.4f}±{f['std']:.4f}")
    return "\n".join(lines)
