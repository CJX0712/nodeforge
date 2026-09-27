"""Training layer (generic loop utilities shared by graph models)."""
from nodeforge.training.trainer import HAS_TORCH, train_full_batch

__all__ = ["HAS_TORCH", "train_full_batch"]
