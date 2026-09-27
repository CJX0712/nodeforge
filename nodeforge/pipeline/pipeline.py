"""NodeForgePipeline: benchmark orchestration.

Determinism contract: with the same config, two runs produce bitwise-identical
benchmark.json except the ``timing_sec`` block.
"""
from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np
import sklearn

from nodeforge.core.config import NodeForgeConfig
from nodeforge.core.errors import PipelineError
from nodeforge.core.seed import set_all
from nodeforge.data.generators import get_benchmark_tiers
from nodeforge.eval.failure_cases import analyze_failures
from nodeforge.eval.metrics import evaluate
from nodeforge.eval.report import aggregate, judge
from nodeforge.graph.factory import AVAILABLE_MODELS, HAS_TORCH, create_model
from nodeforge.preprocess.splits import make_masks

__version__ = "0.1.0"

ABLATION_VARIANTS = {
    "adaprop": dict(),
    "adaprop_nogate": dict(use_edge_gate=False),
    "adaprop_noknn": dict(use_knn=False),
    "adaprop_nores": dict(force_residual=0.0),
}


class NodeForgePipeline:
    def __init__(self, cfg: NodeForgeConfig | None = None) -> None:
        self.cfg = cfg or NodeForgeConfig()

    # ------------------------------------------------------------- single tier
    def run_tier(self, tier: str, seed: int, models: list[str] | None = None) -> dict:
        """Fit every model on one tier/seed; returns {model: {"acc":..,"macro_f1":..}}."""
        tiers = get_benchmark_tiers()
        if tier not in tiers:
            raise PipelineError(f"unknown tier {tier!r}, choose from {sorted(tiers)}")
        set_all(seed)
        bundle = tiers[tier](seed)
        rng = np.random.default_rng(seed * 7919 + 1)
        masks = make_masks(bundle.y, rng, self.cfg.n_train_per_class, self.cfg.n_val)
        out: dict[str, Any] = {}
        for name in models if models is not None else sorted(AVAILABLE_MODELS):
            model = create_model(name, self.cfg, seed)
            model.fit(bundle, masks)
            pred = model.predict(bundle)
            out[name] = evaluate(bundle.y[masks.test], pred[masks.test])
        return out

    # -------------------------------------------------------------- benchmark
    def benchmark(self) -> dict:
        """Full benchmark: tiers x seeds x models + ablation + failure cases + gate."""
        t0 = time.perf_counter()
        cfg = self.cfg
        seeds = [cfg.seed + i for i in range(cfg.n_seeds)]
        tiers = get_benchmark_tiers()
        results: dict[str, dict[str, dict]] = {}
        ablation: dict[str, dict[str, dict]] = {}
        failures: dict[str, list] = {}
        per_seed_raw: dict[str, dict[str, list]] = {}
        abl_raw: dict[str, dict[str, list]] = {}

        for tier in sorted(tiers):
            per_seed_raw[tier] = {m: [] for m in sorted(AVAILABLE_MODELS)}
            abl_raw[tier] = {v: [] for v in ABLATION_VARIANTS}
            for seed in seeds:
                set_all(seed)
                bundle = tiers[tier](seed)
                rng = np.random.default_rng(seed * 7919 + 1)
                masks = make_masks(bundle.y, rng, cfg.n_train_per_class, cfg.n_val)

                fitted: dict[str, tuple] = {}
                for name in sorted(AVAILABLE_MODELS):
                    model = create_model(name, cfg, seed)
                    model.fit(bundle, masks)
                    pred = model.predict(bundle)
                    per_seed_raw[tier][name].append(
                        evaluate(bundle.y[masks.test], pred[masks.test]))
                    fitted[name] = (model, pred)

                # failure cases from the primary model on every tier
                if "adaprop" in fitted:
                    _, pred = fitted["adaprop"]
                    failures[tier] = analyze_failures(bundle, pred, masks)

                # ablation runs only where torch exists (baseline torch models)
                if HAS_TORCH:
                    from nodeforge.graph.models import AdaPropClassifier

                    for variant, kw in ABLATION_VARIANTS.items():
                        m = AdaPropClassifier(cfg, seed, **kw)
                        m.fit(bundle, masks)
                        pred = m.predict(bundle)
                        abl_raw[tier][variant].append(
                            evaluate(bundle.y[masks.test], pred[masks.test]))

            results[tier] = {m: aggregate(v) for m, v in per_seed_raw[tier].items()}
            if HAS_TORCH:
                ablation[tier] = {v: aggregate(r) for v, r in abl_raw[tier].items()}

        gate = judge(results)
        torch_ver = None
        if HAS_TORCH:
            import torch

            torch_ver = torch.__version__
        timing = round(time.perf_counter() - t0, 3)
        return {
            "meta": {
                "system": "NodeForge", "version": __version__, "author": "晨星",
                "python": platform.python_version(), "numpy": np.__version__,
                "sklearn": sklearn.__version__, "torch": torch_ver,
                "has_torch": HAS_TORCH, "seeds": seeds,
                "config": {k: getattr(cfg, k) for k in vars(cfg)},
            },
            "results": results,
            "ablation": ablation,
            "gate": gate,
            "failure_cases": failures,
            "timing_sec": timing,
        }

    # --------------------------------------------------------------------- hpo
    def hpo(self, tier: str, seed: int, budget: int) -> tuple[dict, float]:
        """Budget-limited grid over (hidden, dropout, tau) scored on val F1."""
        from nodeforge.hpo.search import grid_search

        tiers = get_benchmark_tiers()
        if tier not in tiers:
            raise PipelineError(f"unknown tier {tier!r}")
        set_all(seed)
        bundle = tiers[tier](seed)
        rng = np.random.default_rng(seed * 7919 + 1)
        masks = make_masks(bundle.y, rng, self.cfg.n_train_per_class, self.cfg.n_val)

        def score(params: dict) -> float:
            cfg = NodeForgeConfig(**{**vars(self.cfg), **params})
            model = create_model("adaprop", cfg, seed)
            model.fit(bundle, masks)
            pred = model.predict(bundle)
            from nodeforge.eval.metrics import macro_f1

            return macro_f1(bundle.y[masks.val], pred[masks.val])

        return grid_search(
            score,
            {"tau": [0.3, 0.5, 0.8], "knn_k": [8, 15]},
            budget,
        )
