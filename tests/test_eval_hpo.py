import numpy as np
import pytest

from nodeforge.core.errors import EvalError, PipelineError
from nodeforge.data.generators import make_sbm
from nodeforge.eval.failure_cases import analyze_failures
from nodeforge.eval.metrics import accuracy, macro_f1
from nodeforge.eval.report import aggregate, cli_table, judge, significant
from nodeforge.hpo.search import grid_search
from nodeforge.preprocess.splits import make_masks


def test_metrics_correctness():
    y = np.array([0, 0, 1, 1])
    assert accuracy(y, y) == 1.0
    assert macro_f1(y, y) == 1.0
    pred = np.array([0, 1, 1, 1])
    assert accuracy(y, pred) == 0.75
    assert 0.0 < macro_f1(y, pred) < 1.0


def test_metrics_length_mismatch():
    with pytest.raises(EvalError):
        accuracy(np.array([0, 1]), np.array([0]))
    with pytest.raises(EvalError):
        macro_f1(np.array([0, 1]), np.array([0]))


def test_aggregate_stats():
    agg = aggregate([{"acc": 0.5, "macro_f1": 0.4}, {"acc": 0.7, "macro_f1": 0.6}])
    assert agg["acc"]["mean"] == pytest.approx(0.6)
    assert agg["acc"]["std"] == pytest.approx(0.1 * np.sqrt(2))  # ddof=1
    assert len(agg["macro_f1"]["per_seed"]) == 2


def test_significant_gate():
    assert significant(0.9, 0.0, 0.8, 0.0)
    assert not significant(0.9, 0.2, 0.85, 0.2)


def _results(delta, std=0.01):
    def tier(name):
        return {
            "adaprop": {"macro_f1": {"mean": 0.9, "std": std},
                        "acc": {"mean": 0.9, "std": std}},
            "sage": {"macro_f1": {"mean": 0.9 - delta, "std": std},
                     "acc": {"mean": 0.9 - delta, "std": std}},
        }
    return {"t1": tier("t1"), "t2": tier("t2")}


def test_judge_pass_and_fail():
    ok = judge(_results(0.05))
    assert ok["passed"] and ok["significant_tiers"] == 2 and ok["all_positive"]
    bad = judge(_results(0.0))
    assert not bad["passed"]


def test_cli_table_alignment():
    res = _results(0.05)
    text = cli_table(res)
    lines = text.splitlines()
    assert all(len(l) == len(lines[0]) for l in lines)


def test_failure_cases_structure():
    b = make_sbm(50, 0.08, 0.01, seed=9)
    masks = make_masks(b.y, np.random.default_rng(9), 20, 100)
    y_pred = b.y.copy()
    # force three test-node errors
    test_idx = np.flatnonzero(masks.test)[:3]
    y_pred[test_idx] = (y_pred[test_idx] + 1) % b.n_classes
    cases = analyze_failures(b, y_pred, masks, max_cases=3)
    assert len(cases) == 3
    for c in cases:
        assert {"node", "y_true", "y_pred", "reason", "reason_text"} <= set(c)


def test_grid_search_budget_and_best():
    calls = []

    def scorer(params):
        calls.append(params)
        return params["x"]

    best, score = grid_search(scorer, {"x": [1, 5, 3]}, budget=3)
    assert best == {"x": 5} and score == 5 and len(calls) == 3
    with pytest.raises(PipelineError):
        grid_search(scorer, {"x": [1]}, budget=0)
