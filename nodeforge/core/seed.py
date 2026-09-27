"""Global determinism entry point. Everything seeds through here."""
from __future__ import annotations

import random

import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """Seed python random, numpy legacy global RNG and torch (if available).

    Returns a fresh ``numpy.random.Generator`` for downstream generators so that
    data generation is reproducible from a single integer.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # torch is an optional backend (Tier-0); deterministic when present.
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # tiny full-batch graphs: thread contention dominates wall time otherwise
        torch.set_num_threads(min(4, torch.get_num_interop_threads() or 4))
    except (ImportError, RuntimeError):  # pragma: no cover - exercised in fallback tests
        pass
    return np.random.default_rng(seed)
