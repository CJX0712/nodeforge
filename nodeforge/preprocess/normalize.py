"""Adjacency normalization helpers (numpy, shared by torch and fallback paths)."""
from __future__ import annotations

import numpy as np

from nodeforge.core.types import GraphBundle


def symnorm(a: np.ndarray) -> np.ndarray:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2} of a dense matrix."""
    a = a.astype(np.float32, copy=False)
    a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(axis=1)
    deg = np.maximum(deg, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    return (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]).astype(np.float32)


def feature_similarity_weights(bundle: GraphBundle, tau: float) -> np.ndarray:
    """AdaProp edge gating: W_ij = exp(cos(x_i, x_j) / tau) on existing edges."""
    return gated_feature_weights(bundle.x, bundle.adjacency(), tau)


def gated_feature_weights(x: np.ndarray, a: np.ndarray, tau: float) -> np.ndarray:
    """Core gating over an explicit feature matrix (allows train-scaled x)."""
    x = x.astype(np.float32)
    norms = np.maximum(np.linalg.norm(x, axis=1), 1e-12)
    xn = x / norms[:, None]
    cos = xn @ xn.T  # cosine similarity in [-1, 1]
    w = np.exp(cos / tau)
    w[a == 0.0] = 0.0  # gating only re-weights existing edges
    return w.astype(np.float32)
