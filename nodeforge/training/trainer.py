"""Torch training loop shared by all torch models. Full-batch, deterministic.

Layering note: graph/models.py imports this helper; no module imports graph
back from training, so the dependency graph stays acyclic.
"""
from __future__ import annotations

import numpy as np

from nodeforge.core.errors import TrainError

try:  # torch is an optional Tier-0 backend
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in fallback tests
    HAS_TORCH = False

if HAS_TORCH:

    def train_full_batch(
        module: nn.Module,
        inputs: tuple,
        y_t: torch.Tensor,
        train_mask_t: torch.Tensor,
        val_mask_t: torch.Tensor,
        *,
        lr: float,
        weight_decay: float,
        epochs: int,
        patience: int,
        param_groups: list | None = None,
    ) -> dict:
        """Full-batch training with early stopping on val accuracy.

        ``inputs`` is the tuple forwarded to ``module(*inputs)`` (single graph
        tensor for GCN/SAGE/MLP, (x, a_gated, a_plain) for AdaProp).
        ``param_groups`` optionally replaces the flat parameter list (per-group
        lr / weight_decay, e.g. to exempt the AdaProp channel-mix scalar).
        Returns the best state dict plus history. Raises TrainError on NaN loss.
        """
        if param_groups:
            opt = torch.optim.Adam(param_groups)
        else:
            opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
        best_val, best_state, best_epoch = -1.0, None, -1
        history: list[float] = []
        for epoch in range(epochs):
            module.train()
            opt.zero_grad()
            logits = module(*inputs)
            loss = nn.functional.cross_entropy(logits[train_mask_t], y_t[train_mask_t])
            if not torch.isfinite(loss):
                raise TrainError(f"non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            opt.step()
            history.append(float(loss.item()))

            module.eval()
            with torch.no_grad():
                logits = module(*inputs)
                val_acc = float(
                    (logits[val_mask_t].argmax(1) == y_t[val_mask_t]).float().mean().item()
                )
            if val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            if epoch - best_epoch >= patience:
                break
        if best_state is None:  # pragma: no cover - patience >= 1 makes this impossible
            raise TrainError("no best state captured")
        return {"state": best_state, "best_val_acc": best_val, "best_epoch": best_epoch,
                "history": history}


def to_tensor(x: np.ndarray) -> torch.Tensor:
    if not HAS_TORCH:  # pragma: no cover
        raise TrainError("torch is not available")
    return torch.as_tensor(np.ascontiguousarray(x))
