"""Interface protocols. One semantic rule across all models: higher is better,
predict() returns hard labels in [0, n_classes)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from nodeforge.core.types import GraphBundle, SplitMasks


@runtime_checkable
class NodeClassifier(Protocol):
    """Every model (torch GNN, sklearn baseline, numpy fallback) obeys this."""

    name: str

    def fit(self, bundle: GraphBundle, masks: SplitMasks) -> NodeClassifier:
        """Fit on train nodes only; may early-stop on val nodes."""
        ...

    def predict(self, bundle: GraphBundle) -> np.ndarray:
        """Return int64 labels [n] for every node (transductive)."""
        ...
