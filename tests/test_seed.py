import numpy as np
import pytest

from nodeforge.core.seed import set_all


def test_set_all_returns_generator():
    rng = set_all(42)
    assert isinstance(rng, np.random.Generator)


def test_set_all_deterministic_random():
    set_all(7)
    a = [np.random.rand() for _ in range(3)]
    set_all(7)
    b = [np.random.rand() for _ in range(3)]
    assert a == b


def test_set_all_negative_seed_raises():
    with pytest.raises(ValueError):
        set_all(-1)
