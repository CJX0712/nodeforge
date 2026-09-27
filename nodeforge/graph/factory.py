"""Model factory and backend availability."""
from __future__ import annotations

from nodeforge.core.errors import ModelError

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in fallback tests
    HAS_TORCH = False

TORCH_MODELS = ["mlp", "gcn", "sage"]
NUMPY_MODELS = ["logreg", "lp", "netmf", "adaprop", "adaprop_np"]
AVAILABLE_MODELS = NUMPY_MODELS + (TORCH_MODELS if HAS_TORCH else [])


def create_model(name: str, cfg, seed: int):
    """Build a NodeClassifier by name. Raises ModelError (E200) if unknown or
    if a torch model is requested without torch installed."""
    if name in TORCH_MODELS and not HAS_TORCH:
        raise ModelError(f"model {name!r} requires torch, which is not installed")
    if name == "logreg":
        from nodeforge.graph.models import LogRegBaseline
        return LogRegBaseline()
    if name == "lp":
        from nodeforge.graph.models import LabelPropagation
        return LabelPropagation()
    if name == "netmf":
        from nodeforge.graph.models import NetMFLR
        return NetMFLR()
    if name == "adaprop":
        from nodeforge.graph.models import AdaPropClassifier
        return AdaPropClassifier(cfg, seed)
    if name == "adaprop_np":
        from nodeforge.graph.models import NumpyPropClassifier
        return NumpyPropClassifier(tau=cfg.tau)
    if not HAS_TORCH:  # pragma: no cover
        raise ModelError(f"unknown model {name!r}")
    if name == "mlp":
        from nodeforge.graph.models import MLPClassifier
        return MLPClassifier(cfg, seed)
    if name == "gcn":
        from nodeforge.graph.models import GCNClassifier
        return GCNClassifier(cfg, seed)
    if name == "sage":
        from nodeforge.graph.models import SAGEClassifier
        return SAGEClassifier(cfg, seed)
    raise ModelError(f"unknown model {name!r}")
