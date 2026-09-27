"""Failure-case mining: pick misclassified test nodes and auto-attribute causes."""
from __future__ import annotations

import numpy as np

from nodeforge.core.types import GraphBundle, SplitMasks

_REASONS = {
    "low_degree": "low-degree node (<25th percentile): little structural signal",
    "minority_class": "minority class (smallest train support): imbalance pressure",
    "hetero_neighborhood": "heterophilous neighborhood (most neighbors in other classes)",
    "generic": "borderline case: neither extreme degree, class rarity nor heterophily",
}


def analyze_failures(
    bundle: GraphBundle,
    y_pred: np.ndarray,
    masks: SplitMasks,
    max_cases: int = 3,
) -> list[dict]:
    """Return up to ``max_cases`` misclassified test nodes with attribution."""
    deg = bundle.degrees()
    deg_p25 = float(np.percentile(deg, 25))
    classes, counts = np.unique(bundle.y, return_counts=True)
    train_support = {c: int(((bundle.y == c) & masks.train).sum()) for c in classes}
    min_support = min(train_support.values())
    imbalanced = len(classes) > 1 and counts.max() != counts.min()
    # neighborhood label homophily
    n = bundle.num_nodes
    neighbor_homo = np.zeros(n, dtype=np.float64)
    src, dst = bundle.edge_index
    for i in range(n):
        nb = dst[src == i]
        if len(nb):
            neighbor_homo[i] = float((bundle.y[nb] == bundle.y[i]).mean())

    wrong = np.flatnonzero(masks.test & (y_pred != bundle.y))
    cases: list[dict] = []
    for i in wrong:
        attrs = {
            "node": int(i),
            "y_true": int(bundle.y[i]),
            "y_pred": int(y_pred[i]),
            "degree": int(deg[i]),
            "class_train_support": train_support[int(bundle.y[i])],
            "neighbor_homophily": round(float(neighbor_homo[i]), 4),
        }
        if deg[i] < deg_p25:
            attrs["reason"] = "low_degree"
        elif imbalanced and train_support[int(bundle.y[i])] == min_support:
            attrs["reason"] = "minority_class"
        elif neighbor_homo[i] < 0.5:
            attrs["reason"] = "hetero_neighborhood"
        else:
            attrs["reason"] = "generic"
        cases.append(attrs)
        if len(cases) >= max_cases:
            break
    for c in cases:
        c["reason_text"] = _REASONS[c["reason"]]
    return cases
