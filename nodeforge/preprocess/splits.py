"""Stratified split masks. Only train masks are ever used for fitting."""
from __future__ import annotations

import numpy as np

from nodeforge.core.errors import DataError
from nodeforge.core.types import SplitMasks


def make_masks(
    y: np.ndarray,
    rng: np.random.Generator,
    n_train_per_class: int = 20,
    n_val: int = 100,
) -> SplitMasks:
    """Stratified train / val / test masks.

    train: n_train_per_class per class (classic semi-supervised regime).
    val  : n_val nodes drawn from remaining nodes (uniform, seeded).
    test : everything else. Raises DataError if a class is too small.
    """
    n = len(y)
    train = np.zeros(n, dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) <= n_train_per_class:
            raise DataError(f"class {c} has only {len(idx)} nodes")
        pick = rng.choice(idx, size=n_train_per_class, replace=False)
        train[pick] = True
    rest = np.flatnonzero(~train)
    n_val = min(n_val, max(1, len(rest) // 5))
    val_pick = rng.choice(rest, size=n_val, replace=False)
    val = np.zeros(n, dtype=bool)
    val[val_pick] = True
    test = ~(train | val)
    return SplitMasks(train=train, val=val, test=test)
