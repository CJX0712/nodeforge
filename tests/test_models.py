import numpy as np
import pytest

from nodeforge.core.errors import ModelError
from nodeforge.data.generators import make_sbm
from nodeforge.graph.factory import HAS_TORCH, create_model
from nodeforge.preprocess.splits import make_masks


@pytest.fixture(scope="module")
def easy_graph():
    b = make_sbm(50, 0.08, 0.01, seed=7)
    masks = make_masks(b.y, np.random.default_rng(7), 20, 100)
    return b, masks


def _fit_predict(name, b, masks, **kw):
    from nodeforge.core.config import NodeForgeConfig

    cfg = NodeForgeConfig(**kw)
    m = create_model(name, cfg, 42)
    m.fit(b, masks)
    return m.predict(b)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gcn_learns_easy_graph(easy_graph):
    b, masks = easy_graph
    pred = _fit_predict("gcn", b, masks)
    acc = (pred[masks.test] == b.y[masks.test]).mean()
    assert acc > 0.85


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_adaprop_beats_on_hetero():
    b = make_sbm(50, 0.008, 0.05, feat_noise=1.6, seed=7)
    masks = make_masks(b.y, np.random.default_rng(7), 20, 100)
    pred = _fit_predict("adaprop", b, masks)
    acc = (pred[masks.test] == b.y[masks.test]).mean()
    assert acc > 0.80


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_adaprop_variants_construct(easy_graph):
    b, masks = easy_graph
    from nodeforge.core.config import NodeForgeConfig

    cfg = NodeForgeConfig()
    for kw in (dict(use_edge_gate=False), dict(use_residual=False)):
        m = create_model("adaprop", cfg, 42)
        m.use_edge_gate = kw.get("use_edge_gate", True)
        m.use_residual = kw.get("use_residual", True)
        m.fit(b, masks)
        pred = m.predict(b)
        assert pred.shape == (b.num_nodes,)


def test_unknown_model_raises():
    from nodeforge.core.config import NodeForgeConfig

    with pytest.raises(ModelError):
        create_model("nope", NodeForgeConfig(), 0)


def test_lp_and_logreg_numpy_paths(easy_graph):
    b, masks = easy_graph
    for name in ("logreg", "lp"):
        pred = _fit_predict(name, b, masks)
        acc = (pred[masks.test] == b.y[masks.test]).mean()
        assert acc > 0.60, name


def test_numpy_fallback_runs(easy_graph):
    b, masks = easy_graph
    pred = _fit_predict("adaprop_np", b, masks)
    acc = (pred[masks.test] == b.y[masks.test]).mean()
    assert acc > 0.60
