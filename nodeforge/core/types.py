"""Dataclasses shared across the package. Numpy arrays only (no framework types)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nodeforge.core.errors import DataError


@dataclass
class GraphBundle:
    """A node-classification graph in plain numpy form.

    x          : float32 [n, d] node features
    edge_index : int64   [2, m] undirected edges, both directions present
    y          : int64   [n]    labels in [0, n_classes)
    """

    x: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    n_classes: int
    name: str = "graph"

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        if self.x.ndim != 2:
            raise DataError(f"x must be 2-D [n, d], got shape {self.x.shape}")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise DataError("edge_index must be 2-D [2, m]")
        if self.y.shape[0] != n:
            raise DataError(f"y length {self.y.shape[0]} != n nodes {n}")
        if self.edge_index.min() < 0 or self.edge_index.max() >= n:
            raise DataError("edge_index out of node range")
        if self.y.min() < 0 or self.y.max() >= self.n_classes:
            raise DataError("y out of class range")
        if not np.issubdtype(self.y.dtype, np.integer):
            self.y = self.y.astype(np.int64)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def feat_dim(self) -> int:
        return int(self.x.shape[1])

    def adjacency(self) -> np.ndarray:
        """Dense symmetric adjacency [n, n] float32 without self-loops."""
        n = self.num_nodes
        a = np.zeros((n, n), dtype=np.float32)
        src, dst = self.edge_index[0], self.edge_index[1]
        a[src, dst] = 1.0
        a[dst, src] = 1.0
        np.fill_diagonal(a, 0.0)
        return a

    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.num_nodes, dtype=np.int64)
        np.add.at(deg, self.edge_index[0], 1)
        return deg


@dataclass
class SplitMasks:
    """Boolean masks over nodes. Train/val/test are disjoint and cover all nodes."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        if (self.train & self.val).any() or (self.train & self.test).any() or (self.val & self.test).any():
            raise DataError("masks must be disjoint")
        if not (self.train | self.val | self.test).all():
            raise DataError("masks must cover every node")
        if not self.train.any():
            raise DataError("train mask is empty")
        if not self.test.any():
            raise DataError("test mask is empty")
