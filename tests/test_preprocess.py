import numpy as np
import pytest

from nodeforge.core.errors import DataError
from nodeforge.data.generators import make_sbm
from nodeforge.preprocess.normalize import feature_similarity_weights, symnorm
from nodeforge.preprocess.splits import make_masks


def test_symnorm_properties():
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    s = symnorm(a)
    assert np.allclose(s, s.T)
    assert (s > 0).any() and np.isfinite(s).all()
    # symmetric normalization keeps row masses in a sane band
    assert (s.sum(axis=1) >= 0.9).all() and (s.sum(axis=1) <= 2.1).all()


def test_feature_weights_zero_off_edges():
    b = make_sbm(30, 0.08, 0.01, seed=1)
    w = feature_similarity_weights(b, tau=0.5)
    a = b.adjacency()
    assert (w[a == 0] == 0).all()
    assert (w[a == 1] > 0).any()


def test_masks_stratified_and_disjoint():
    b = make_sbm(50, 0.08, 0.01, seed=2)
    rng = np.random.default_rng(0)
    m = make_masks(b.y, rng, n_train_per_class=20, n_val=100)
    assert m.train.sum() == 20 * b.n_classes
    assert not (m.train & m.val).any()
    assert not (m.train & m.test).any()
    assert (m.train | m.val | m.test).all()
    for c in range(b.n_classes):
        assert (m.train & (b.y == c)).sum() == 20


def test_masks_class_too_small():
    b = make_sbm(10, 0.08, 0.01, seed=2)  # 60 nodes, class size 10
    rng = np.random.default_rng(0)
    with pytest.raises(DataError):
        make_masks(b.y, rng, n_train_per_class=20, n_val=10)
