import numpy as np
import pytest

from nodeforge.core.errors import DataError
from nodeforge.data.generators import get_benchmark_tiers, make_lfr, make_sbm


def test_sbm_shapes_and_reproducible():
    b1 = make_sbm(50, 0.08, 0.01, seed=3)
    b2 = make_sbm(50, 0.08, 0.01, seed=3)
    assert b1.num_nodes == 300
    assert (b1.edge_index == b2.edge_index).all()
    assert (b1.x == b2.x).all()
    assert b1.n_classes == 6


def test_sbm_heterophilous_labels():
    b = make_sbm(50, 0.01, 0.08, seed=5)
    a = b.adjacency()
    same = 0.0
    total = 0
    for i in range(b.num_nodes):
        nb = np.flatnonzero(a[i])
        same += (b.y[nb] == b.y[i]).sum()
        total += len(nb)
    assert same / total < 0.2  # edges mostly cross-class


def test_sbm_invalid_params():
    with pytest.raises(DataError):
        make_sbm(5, 0.1, 0.1, seed=0)  # n_per_class too small
    with pytest.raises(DataError):
        make_sbm(50, 1.5, 0.1, seed=0)  # p_in out of range


def test_lfr_reproducible_and_labeled():
    b1 = make_lfr(n=500, mu=0.3, seed=11)
    b2 = make_lfr(n=500, mu=0.3, seed=11)
    assert b1.num_nodes == 500
    assert (b1.edge_index == b2.edge_index).all()
    assert (b1.y == b2.y).all()
    assert b1.n_classes >= 2


def test_lfr_invalid_mu():
    with pytest.raises(DataError):
        make_lfr(mu=1.5, seed=0)


def test_benchmark_tiers_registry():
    tiers = get_benchmark_tiers()
    assert len(tiers) == 3
    b = tiers["tier3_heterophilous"](0)
    assert b.num_nodes == 600 and b.n_classes == 6
