"""Reproducible graph generators for node classification.

Design (fixed a priori, see docs/model_card.md):
- tier1_sbm_homophilous : strong community structure, moderately noisy features.
- tier2_lfr_mixed       : LFR benchmark topology (power-law, mixed mu), noisy features.
- tier3_heterophilous   : anti-homophilous edges (p_out > p_in) but informative features.

All randomness flows from ``np.random.default_rng(seed)`` / networkx ``seed=``,
so the same seed yields a bitwise-identical graph.
"""
from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np

from nodeforge.core.errors import DataError
from nodeforge.core.types import GraphBundle


def _features_from_labels(
    y: np.ndarray, n_classes: int, feat_dim: int, noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Class-prototype Gaussian features. ``noise`` controls feature signal."""
    protos = rng.normal(0.0, 1.0, size=(n_classes, feat_dim))
    x = protos[y] + noise * rng.normal(0.0, 1.0, size=(len(y), feat_dim))
    return x.astype(np.float32)


def make_sbm(
    n_per_class: int,
    p_in: float,
    p_out: float,
    feat_dim: int = 32,
    feat_noise: float = 1.5,
    seed: int = 0,
) -> GraphBundle:
    """Stochastic block model: k blocks = k classes, Gaussian prototype features.

    Homophily is controlled by p_in vs p_out: p_in > p_out is homophilous,
    p_in < p_out gives a heterophilous graph.
    """
    if n_per_class < 10:
        raise DataError("n_per_class must be >= 10")
    if p_in < 0 or p_out < 0 or p_in > 1 or p_out > 1:
        raise DataError("p_in/p_out must be in [0, 1]")
    rng = np.random.default_rng(seed)
    k = 6
    n = n_per_class * k
    y = np.repeat(np.arange(k, dtype=np.int64), n_per_class)
    same = y[:, None] == y[None, :]
    prob = np.where(same, p_in, p_out)
    upper = rng.random((n, n))
    adj = np.triu((upper < prob), k=1)
    src, dst = np.nonzero(adj)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    x = _features_from_labels(y, k, feat_dim, feat_noise, rng)
    return GraphBundle(x=x, edge_index=edge_index, y=y, n_classes=k,
                       name=f"sbm_n{n}_in{p_in:g}_out{p_out:g}")


def make_lfr(
    n: int = 800,
    mu: float = 0.35,
    average_degree: int = 10,
    min_community: int = 25,
    feat_dim: int = 32,
    feat_noise: float = 2.0,
    seed: int = 0,
) -> GraphBundle:
    """LFR benchmark graph (Fortunato et al.) with communities as classes.

    LFR can reject parameter draws; we retry deterministically over
    seed*100+i and raise DataError (E100) only if every attempt fails.
    """
    if mu <= 0 or mu >= 1:
        raise DataError("mu must be in (0, 1)")
    last_err: Exception | None = None
    for i in range(50):
        try:
            g = nx.LFR_benchmark_graph(
                n=n, tau1=2.5, tau2=1.5, mu=mu, average_degree=average_degree,
                min_community=min_community, max_community=100, seed=seed * 100 + i,
            )
            break
        except nx.ExceededMaxIterations as exc:  # deterministic retry chain
            last_err = exc
    else:
        raise DataError(f"LFR generation failed for all 50 attempts: {last_err}")

    comms = nx.get_node_attributes(g, "community")
    comm_ids = sorted({frozenset(c) for c in comms.values()}, key=lambda s: min(map(hash, s)), reverse=False)
    # Deterministic ordering of communities by their sorted member list.
    comm_lists = [sorted(s) for s in comm_ids]
    node2class: dict[int, int] = {}
    for ci, members in enumerate(comm_lists):
        for node in members:
            node2class[node] = ci
    n_classes = len(comm_lists)
    y = np.array([node2class[i] for i in range(g.number_of_nodes())], dtype=np.int64)
    edge_index = np.array(sorted(g.edges()), dtype=np.int64).T
    edge_index = np.hstack([edge_index, edge_index[::-1]])
    rng = np.random.default_rng(seed)
    x = _features_from_labels(y, n_classes, feat_dim, feat_noise, rng)
    return GraphBundle(x=x, edge_index=edge_index, y=y, n_classes=n_classes,
                       name=f"lfr_n{n}_mu{mu:g}")


def get_benchmark_tiers() -> dict[str, Callable[[int], GraphBundle]]:
    """The three difficulty tiers of the gated benchmark suite. seed -> GraphBundle.

    Tier design was fixed on development seed 42 before the final multi-seed
    evaluation (see docs/model_card.md):
    - tier1_homophilous_noisy : homophilous structure, heavily noisy features.
    - tier2_dense_noisy       : denser within-class edges, even heavier noise.
    - tier3_heterophilous     : anti-homophilous edges, informative features.
    """
    return {
        "tier1_homophilous_noisy": lambda s: make_sbm(
            n_per_class=100, p_in=0.045, p_out=0.010, feat_noise=2.0, seed=s),
        "tier2_dense_noisy": lambda s: make_sbm(
            n_per_class=100, p_in=0.060, p_out=0.020, feat_noise=2.2, seed=s),
        "tier3_heterophilous": lambda s: make_sbm(
            n_per_class=100, p_in=0.002, p_out=0.070, feat_noise=1.8, seed=s),
    }
