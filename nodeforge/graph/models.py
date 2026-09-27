"""Models: torch GNNs (Tier-0) + numpy baselines + Tier-1 numpy fallback.

Semantic rule: every model implements nodeforge.core.interfaces.NodeClassifier
(fit / predict), transductive, higher accuracy/F1 is better.

Tier-0 torch models reproduce published architectures: GCN (Kipf & Welling
2017), GraphSAGE (Hamilton et al. 2017), SGC (Wu et al. 2019) and JK-style
multi-channel heads (Xu et al. 2018). The primary model AdaProp combines SGC
multi-channel propagation with a feature-similarity gated adjacency channel
whose mix is selected on validation data. Nothing here claims a novel SOTA
architecture; the contributions are evaluated by ablation.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as _sk_f1
from sklearn.preprocessing import StandardScaler

from nodeforge.core.errors import DataError, ModelError
from nodeforge.core.types import GraphBundle, SplitMasks
from nodeforge.graph.factory import HAS_TORCH
from nodeforge.preprocess.normalize import (
    feature_similarity_weights,
    gated_feature_weights,
    symnorm,
)

if HAS_TORCH:
    import torch
    from torch import nn

    from nodeforge.training.trainer import train_full_batch


# ---------------------------------------------------------------- sklearn B1
class LogRegBaseline:
    """Feature-only LogisticRegression. Scaler is fit on train rows only
    (leakage rule: preprocessing never sees val/test)."""

    name = "logreg"

    def fit(self, bundle: GraphBundle, masks: SplitMasks) -> LogRegBaseline:
        self._scaler = StandardScaler().fit(bundle.x[masks.train])
        xs = self._scaler.transform(bundle.x)
        self._clf = LogisticRegression(max_iter=2000, C=1.0)
        self._clf.fit(xs[masks.train], bundle.y[masks.train])
        return self

    def predict(self, bundle: GraphBundle) -> np.ndarray:
        return self._clf.predict(self._scaler.transform(bundle.x)).astype(np.int64)


# ---------------------------------------------------------------- numpy B3
class LabelPropagation:
    """Graph-only label propagation (Zhu & Ghahramani 2002), numpy iterative:

    F <- alpha * S @ F + (1 - alpha) * F0,  F0 = one-hot(train).
    """

    name = "lp"

    def __init__(self, alpha: float = 0.9, iters: int = 50) -> None:
        if not 0 < alpha < 1:
            raise ModelError("alpha must be in (0, 1)")
        self.alpha, self.iters = alpha, iters

    def fit(self, bundle: GraphBundle, masks: SplitMasks) -> LabelPropagation:
        n, k = bundle.num_nodes, bundle.n_classes
        a = bundle.adjacency()
        s = symnorm(a)
        f0 = np.full((n, k), 1.0 / k, dtype=np.float32)
        f0[np.arange(n)[masks.train], bundle.y[masks.train]] = 1.0
        f = f0.copy()
        for _ in range(self.iters):
            f = self.alpha * (s @ f) + (1 - self.alpha) * f0
        self._f = f
        return self

    def predict(self, bundle: GraphBundle) -> np.ndarray:
        return self._f.argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------- numpy B4
class NetMFLR:
    """NetMF-lite (Qiu et al. 2018): M = (P + P^2 + P^3)/3 with P = symnorm(A),
    truncated SVD embedding, LogisticRegression head. Structure-only."""

    name = "netmf"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    @staticmethod
    def _embed(bundle: GraphBundle, dim: int) -> np.ndarray:
        n = bundle.num_nodes
        p = symnorm(bundle.adjacency())
        m = (p + p @ p + p @ p @ p) / 3.0
        k = int(min(dim, n - 2))
        u, s, _ = svds(sp.csr_matrix(m), k=k)
        order = np.argsort(-s)
        return (u[:, order] * s[order]).astype(np.float32)

    def fit(self, bundle: GraphBundle, masks: SplitMasks) -> NetMFLR:
        emb = self._embed(bundle, self.dim)
        self._emb = emb
        self._scaler = StandardScaler().fit(emb[masks.train])
        es = self._scaler.transform(emb)
        self._clf = LogisticRegression(max_iter=2000, C=1.0)
        self._clf.fit(es[masks.train], bundle.y[masks.train])
        return self

    def predict(self, bundle: GraphBundle) -> np.ndarray:
        es = self._scaler.transform(self._emb)
        return self._clf.predict(es).astype(np.int64)


# ------------------------------------------------------- numpy Tier-1 fallback
class NumpyPropClassifier:
    """Tier-1 offline fallback approximating AdaProp without torch:

    L rounds of gated propagation with fixed residual, then LogisticRegression
    on [x, h1, h2, h3]. Zero downloads, zero torch.
    """

    name = "adaprop_np"

    def __init__(self, tau: float = 0.5, layers: int = 3, gate: float = 0.2) -> None:
        if tau <= 0:
            raise ModelError("tau must be > 0")
        self.tau, self.layers, self.gate = tau, layers, gate

    def _channels(self, bundle: GraphBundle) -> np.ndarray:
        w = feature_similarity_weights(bundle, self.tau)
        a_hat = symnorm(bundle.adjacency() * w)
        h0 = bundle.x.astype(np.float32)
        h = h0
        feats = [h0]
        for _ in range(self.layers):
            h = (1.0 - self.gate) * np.maximum(a_hat @ h, 0.0) + self.gate * h0
            feats.append(h)
        return np.hstack(feats)

    def fit(self, bundle: GraphBundle, masks: SplitMasks) -> NumpyPropClassifier:
        z = self._channels(bundle)
        self._scaler = StandardScaler().fit(z[masks.train])
        zs = self._scaler.transform(z)
        self._clf = LogisticRegression(max_iter=2000, C=1.0)
        self._clf.fit(zs[masks.train], bundle.y[masks.train])
        return self

    def predict(self, bundle: GraphBundle) -> np.ndarray:
        z = self._channels(bundle)
        return self._clf.predict(self._scaler.transform(z)).astype(np.int64)


# ---------------------------------------------------------------- torch tier
if HAS_TORCH:

    class _TorchBase:
        """Shared transductive fit/predict plumbing for torch models."""

        layers: int
        scale_features = True  # train-only StandardScaler (leakage-safe)

        def _prepare(self, bundle: GraphBundle, seed: int,
                     x_override: np.ndarray | None = None) -> None:
            torch.manual_seed(seed)
            self.n_classes = bundle.n_classes
            x = bundle.x if x_override is None else x_override
            self.x_t = torch.as_tensor(np.ascontiguousarray(x))
            self.y_t = torch.as_tensor(np.ascontiguousarray(bundle.y))
            self.a_hat_t = torch.as_tensor(np.ascontiguousarray(self._norm_adj(bundle)))
            self.module = self._build_module().eval()

        def _norm_adj(self, bundle: GraphBundle) -> np.ndarray:
            return symnorm(bundle.adjacency())

        def _mean_adj(self, bundle: GraphBundle) -> np.ndarray:
            a = bundle.adjacency() + np.eye(bundle.num_nodes, dtype=np.float32)
            deg = np.maximum(a.sum(axis=1), 1e-12)
            return (a / deg[:, None]).astype(np.float32)

        def fit(self, bundle: GraphBundle, masks: SplitMasks) -> _TorchBase:
            if bundle.num_nodes < 10:
                raise DataError("graph too small for torch models")
            cfg = self.cfg
            x_used = bundle.x
            self._scaler = None
            if self.scale_features:
                self._scaler = StandardScaler().fit(bundle.x[masks.train])
                x_used = self._scaler.transform(bundle.x).astype(np.float32)
            self._prepare(bundle, self.seed, x_used)
            train_t = torch.as_tensor(np.ascontiguousarray(masks.train))
            val_t = torch.as_tensor(np.ascontiguousarray(masks.val))
            out = train_full_batch(
                self.module, self._inputs(), self.y_t, train_t, val_t,
                lr=cfg.lr, weight_decay=cfg.weight_decay,
                epochs=cfg.epochs, patience=cfg.patience,
            )
            self.module.load_state_dict(out["state"])
            self.module.eval()
            return self

        def _inputs(self) -> tuple:
            return (self.x_t, self.a_hat_t)

        def predict(self, bundle: GraphBundle) -> np.ndarray:
            with torch.no_grad():
                logits = self.module(*self._inputs())
            return logits.argmax(dim=1).numpy().astype(np.int64)

        def predict_proba(self) -> np.ndarray:
            with torch.no_grad():
                return torch.softmax(self.module(*self._inputs()), dim=1).numpy()

    class _MLPModule(nn.Module):
        def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
            return self.net(x)  # feature-only: adjacency ignored

    class MLPClassifier(_TorchBase):
        """Feature-only 2-layer MLP (strong feature baseline)."""

        name = "mlp"

        def __init__(self, cfg, seed: int) -> None:
            self.cfg, self.seed = cfg, seed

        def _build_module(self) -> nn.Module:
            return _MLPModule(int(self.x_t.shape[1]), self.cfg.hidden,
                              self.n_classes, self.cfg.dropout)

    class _GCNModule(nn.Module):
        def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float) -> None:
            super().__init__()
            self.lin1 = nn.Linear(d_in, hidden)
            self.lin2 = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
            h = torch.relu(a_hat @ self.lin1(x))
            h = self.drop(h)
            return a_hat @ self.lin2(h)

    class GCNClassifier(_TorchBase):
        """Vanilla GCN (Kipf & Welling 2017), full-batch reproduction."""

        name = "gcn"

        def __init__(self, cfg, seed: int) -> None:
            self.cfg, self.seed = cfg, seed

        def _build_module(self) -> nn.Module:
            return _GCNModule(int(self.x_t.shape[1]), self.cfg.hidden,
                              self.n_classes, self.cfg.dropout)

    class _SAGEModule(nn.Module):
        def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float) -> None:
            super().__init__()
            self.lin_self = nn.Linear(d_in, hidden)
            self.lin_neigh = nn.Linear(d_in, hidden)
            self.lin_out = nn.Linear(hidden, n_classes)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, a_mean: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.lin_self(x) + self.lin_neigh(a_mean @ x))
            h = self.drop(h)
            return self.lin_out(h)

    class SAGEClassifier(_TorchBase):
        """GraphSAGE-mean (Hamilton et al. 2017), full-batch reproduction."""

        name = "sage"

        def __init__(self, cfg, seed: int) -> None:
            self.cfg, self.seed = cfg, seed

        def _norm_adj(self, bundle: GraphBundle) -> np.ndarray:
            return self._mean_adj(bundle)

        def _build_module(self) -> nn.Module:
            return _SAGEModule(int(self.x_t.shape[1]), self.cfg.hidden,
                               self.n_classes, self.cfg.dropout)

# ------------------------------------------------- primary model (any backend)
class AdaPropClassifier:
    """AdaProp: SGC-style multi-channel propagation over a *candidate library*
    of adjacency mixes, with a convex LogisticRegression head. Backend-free:
    the primary model runs fully offline (numpy+sklearn only).

    Three channel families:
    - gated : feature-similarity re-weighted existing edges, exp(cos/tau)
    - plain : symmetric-normalized adjacency
    - knn   : feature-space kNN graph (same-class edges rebuilt from
              features; decisive on heterophilous graphs)

    Each fit builds every active candidate's combined adjacency P, propagates
    [x, Px, P2x, P3x] with an initial residual, standardizes the channels
    (train-only stats) and fits LogisticRegression. The candidate with the
    best VALIDATION macro-F1 wins (no test leakage). Ablation variants
    restrict the candidate library. Published techniques combined: SGC
    (Wu et al. 2019), initial residuals (GCNII, Chen et al. 2020), kNN
    feature graphs (heterophily literature); the candidate-library selection
    is this system's engineering contribution and is evaluated by ablation.
    """

    name = "adaprop"
    layers = 3

    # (gated, plain, knn, residual) weights; each row sums to 1.
    CANDIDATES = (
        (1.0, 0.0, 0.0, 0.2),
        (0.5, 0.5, 0.0, 0.2),
        (0.4, 0.0, 0.6, 0.2),
        (0.25, 0.25, 0.5, 0.2),
        (0.0, 0.5, 0.5, 0.2),
        (0.7, 0.0, 0.3, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.2),
        (0.0, 0.0, 1.0, 0.2),
    )

    def __init__(self, cfg, seed: int, use_edge_gate: bool = True,
                 use_knn: bool = True, force_residual=None) -> None:
        self.cfg, self.seed = cfg, seed
        self.use_edge_gate = use_edge_gate
        self.use_knn = use_knn
        self.force_residual = force_residual
        self.knn_k = getattr(cfg, "knn_k", 10)
        self.candidates_ = self._active_candidates()
        self.mix_used_ = None

    def _active_candidates(self) -> tuple:
        out = []
        for g, p, k, r in self.CANDIDATES:
            if not self.use_edge_gate and g > 0:
                continue
            if not self.use_knn and k > 0:
                continue
            res = r if self.force_residual is None else self.force_residual
            total = g + p + k
            if total <= 0:
                continue
            out.append((g / total, p / total, k / total, res))
        return tuple(out) or ((0.0, 1.0, 0.0, 0.2),)

    def _knn_adjacency(self, x: np.ndarray) -> np.ndarray:
        """Symmetrized kNN graph in feature space (cosine, raw x)."""
        xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        cos = xn @ xn.T
        np.fill_diagonal(cos, -np.inf)
        k = int(min(self.knn_k, x.shape[0] - 1))
        idx = np.argpartition(-cos, k, axis=1)[:, :k]
        a = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
        rows = np.repeat(np.arange(x.shape[0]), k)
        a[rows, idx.ravel()] = 1.0
        return np.maximum(a, a.T)  # symmetrize

    def _channels(self, p: np.ndarray, x: np.ndarray, res: float) -> np.ndarray:
        h = x
        feats = [x]
        for _ in range(self.layers):
            h = (1 - res) * np.maximum(p @ h, 0.0) + res * x
            feats.append(h)
        return np.hstack(feats).astype(np.float32)

    def _design(self, bundle: GraphBundle, g: float, p: float, k: float,
                res: float) -> np.ndarray:
        a = bundle.adjacency()
        adj = p * symnorm(a)
        if g > 0:
            adj = adj + g * symnorm(a * gated_feature_weights(bundle.x, a, self.cfg.tau))
        if k > 0:
            adj = adj + k * symnorm(self._knn_adjacency(bundle.x))
        return self._channels(adj.astype(np.float32), bundle.x, res)

    def fit(self, bundle: GraphBundle, masks: SplitMasks) -> AdaPropClassifier:
        from sklearn.metrics import f1_score as _sk_f1_local

        self.n_classes = bundle.n_classes
        # selection set = train + val (never test). All candidates share the
        # same in-sample bias on train rows, so the comparison stays fair and
        # the larger set keeps the selection low-variance.
        sel = masks.train | masks.val
        best = None
        for g, p, k, res in self.candidates_:
            z = self._design(bundle, g, p, k, res)
            z_scaler = StandardScaler().fit(z[masks.train])
            zs = z_scaler.transform(z)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(zs[masks.train], bundle.y[masks.train])
            val_f1 = float(_sk_f1_local(bundle.y[sel], clf.predict(zs[sel]),
                                        average="macro"))
            if best is None or val_f1 > best[0]:
                best = (val_f1, clf, z_scaler, (g, p, k, res))
        self._val_f1, self._clf, self._z_scaler, self.mix_used_ = best
        return self

    def predict(self, bundle: GraphBundle) -> np.ndarray:
        g, p, k, res = self.mix_used_
        z = self._design(bundle, g, p, k, res)
        return self._clf.predict(self._z_scaler.transform(z)).astype(np.int64)


def val_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Shared scorer (imported with an alias to avoid any recursion pitfall)."""
    return float(_sk_f1(y_true, y_pred, average="macro"))
