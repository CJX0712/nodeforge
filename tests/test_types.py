import numpy as np
import pytest

from nodeforge.core.errors import DataError
from nodeforge.core.types import GraphBundle, SplitMasks


def make_bundle():
    x = np.random.rand(6, 3).astype(np.float32)
    src = [0, 1, 2, 3, 4, 5]
    dst = [1, 2, 3, 4, 5, 0]
    ei = np.array([src + dst, dst + src])  # both directions, [2, 12]
    y = np.array([0, 0, 1, 1, 2, 2])
    return GraphBundle(x=x, edge_index=ei, y=y, n_classes=3)


def test_graph_bundle_valid():
    b = make_bundle()
    assert b.num_nodes == 6 and b.num_edges == 12 and b.feat_dim == 3


def test_adjacency_symmetric_no_selfloops():
    b = make_bundle()
    a = b.adjacency()
    assert (a == a.T).all()
    assert (np.diag(a) == 0).all()
    assert a[0, 1] == 1.0 and a[1, 0] == 1.0


def test_degrees():
    b = make_bundle()
    assert (b.degrees() == 2).all()


def test_bad_x_dims():
    with pytest.raises(DataError):
        GraphBundle(x=np.zeros(5), edge_index=np.zeros((2, 2), dtype=int),
                    y=np.zeros(5, dtype=int), n_classes=1)


def test_bad_edge_range():
    with pytest.raises(DataError):
        GraphBundle(x=np.zeros((3, 2), dtype=np.float32),
                    edge_index=np.array([[0], [9]]),
                    y=np.zeros(3, dtype=int), n_classes=1)


def test_masks_disjoint_and_cover():
    t = np.array([True, True, False, False, False, False])
    v = np.array([False, False, True, True, False, False])
    s = np.array([False, False, False, False, True, True])
    m = SplitMasks(train=t, val=v, test=s)
    assert m.train.sum() == 2 and m.test.sum() == 2


def test_masks_overlap_raises():
    t = np.array([True, True, False, False, False, False])
    v = np.array([True, False, True, False, False, False])
    s = np.array([False, False, False, False, True, True])
    with pytest.raises(DataError):
        SplitMasks(train=t, val=v, test=s)


def test_masks_incomplete_raises():
    t = np.array([True, True, False, False, False, False])
    v = np.array([False, False, True, False, False, False])
    s = np.array([False, False, False, False, True, True])
    with pytest.raises(DataError):
        SplitMasks(train=t, val=v, test=s)
