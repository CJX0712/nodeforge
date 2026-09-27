import json

import pytest

from nodeforge.core.errors import PipelineError
from nodeforge.graph.factory import HAS_TORCH
from nodeforge.pipeline.pipeline import NodeForgePipeline


def test_run_tier_unknown_tier():
    pipe = NodeForgePipeline()
    with pytest.raises(PipelineError):
        pipe.run_tier("nope", 42, models=["logreg"])


def test_run_tier_two_seeds_identical():
    """Determinism: same seed -> bitwise identical metrics."""
    pipe = NodeForgePipeline()
    models = ["logreg", "lp"]
    a = pipe.run_tier("tier1_homophilous_noisy", 42, models=models)
    b = pipe.run_tier("tier1_homophilous_noisy", 42, models=models)
    assert a.keys() == b.keys()
    for m in models:
        assert a[m]["acc"] == b[m]["acc"]
        assert a[m]["macro_f1"] == b[m]["macro_f1"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_benchmark_deterministic_core_metrics(tmp_path):
    """Full benchmark twice: core metrics bitwise identical (timing may differ)."""
    pipe = NodeForgePipeline()
    r1 = pipe.benchmark()
    r2 = pipe.benchmark()
    for section in ("results", "ablation"):
        assert json.dumps(r1[section], sort_keys=True) == json.dumps(r2[section], sort_keys=True)
    assert json.dumps(r1["gate"], sort_keys=True) == json.dumps(r2["gate"], sort_keys=True)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_benchmark_structure_and_gate_presence():
    report = NodeForgePipeline().benchmark()
    assert set(report) >= {"meta", "results", "ablation", "gate", "failure_cases"}
    assert len(report["results"]) == 3
    for tier, models in report["results"].items():
        assert "adaprop" in models and "sage" in models
        assert report["gate"]["per_tier"][tier]["delta"] is not None
    # failure cases: at least one per tier, at most three
    for tier, cases in report["failure_cases"].items():
        assert 0 < len(cases) <= 3


def test_hpo_smoke():
    pipe = NodeForgePipeline()
    params, score = pipe.hpo("tier1_homophilous_noisy", 42, budget=2)
    assert params in ({"knn_k": 8, "tau": 0.3}, {"knn_k": 8, "tau": 0.5})
    assert 0.0 <= score <= 1.0
