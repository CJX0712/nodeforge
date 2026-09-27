"""Preprocess layer: splits and adjacency normalization. Leakage-safe by design:
split masks are drawn once per seed and never touch features."""
from nodeforge.preprocess.normalize import (
    feature_similarity_weights,
    gated_feature_weights,
    symnorm,
)
from nodeforge.preprocess.splits import make_masks

__all__ = ["feature_similarity_weights", "gated_feature_weights", "make_masks", "symnorm"]
