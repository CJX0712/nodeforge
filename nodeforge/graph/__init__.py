"""Graph/domain layer: torch GNNs, numpy baselines, Tier-1 fallback."""
from nodeforge.graph.factory import AVAILABLE_MODELS, HAS_TORCH, create_model

__all__ = ["AVAILABLE_MODELS", "HAS_TORCH", "create_model"]
