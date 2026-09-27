"""Runtime configuration with ENV_NODEFORGE_* overrides and schema validation.

Every field can be overridden via environment variables, e.g.
``ENV_NODEFORGE_SEED=7``. Invalid values raise PipelineError (E500).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

from nodeforge.core.errors import PipelineError

_ENV_PREFIX = "ENV_NODEFORGE_"

# Preset performance gate (fixed a priori, see docs/model_card.md).
F1_GATE = 0.03          # mean Macro-F1 advantage required over best baseline
SIGNIFICANT_FRAC = 2    # at least this many tiers must pass the std-based test


@dataclass
class NodeForgeConfig:
    seed: int = 42
    n_seeds: int = 3
    hidden: int = 32
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 150
    patience: int = 30
    tau: float = 0.5              # AdaProp feature-similarity temperature
    knn_k: int = 10               # AdaProp feature-space kNN channel size
    n_train_per_class: int = 20
    n_val: int = 100
    hpo_budget: int = 0           # 0 = disabled (demo stays fast)
    out_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise PipelineError("seed must be >= 0")
        if self.n_seeds < 1:
            raise PipelineError("n_seeds must be >= 1")
        if self.hidden < 4:
            raise PipelineError("hidden must be >= 4")
        if not 0.0 <= self.dropout < 1.0:
            raise PipelineError("dropout must be in [0, 1)")
        if self.lr <= 0:
            raise PipelineError("lr must be > 0")
        if self.epochs < 1 or self.patience < 1:
            raise PipelineError("epochs/patience must be >= 1")
        if self.tau <= 0:
            raise PipelineError("tau must be > 0")
        if self.knn_k < 2:
            raise PipelineError("knn_k must be >= 2")
        if self.n_train_per_class < 1 or self.n_val < 0:
            raise PipelineError("invalid split sizes")
        if self.hpo_budget < 0:
            raise PipelineError("hpo_budget must be >= 0")

    @classmethod
    def from_env(cls) -> NodeForgeConfig:
        overrides: dict[str, object] = {}
        types = {f.name: f.type for f in fields(cls)}
        for key, value in os.environ.items():
            if not key.startswith(_ENV_PREFIX):
                continue
            name = key[len(_ENV_PREFIX):].lower()
            if name not in types:
                raise PipelineError(f"unknown config override: {key}")
            raw = types[name]
            try:
                if raw in ("int", int):
                    overrides[name] = int(value)
                elif raw in ("float", float):
                    overrides[name] = float(value)
                else:
                    overrides[name] = value
            except ValueError as exc:
                raise PipelineError(f"bad value for {key}: {value!r}") from exc
        return cls(**overrides)  # type: ignore[arg-type]
