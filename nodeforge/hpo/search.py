"""Tiny deterministic grid search (val macro-F1). Disabled by default so the
demo stays inside its performance budget; enable with --hpo."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from nodeforge.core.errors import PipelineError


def grid_search(
    evaluate_config: Callable[[dict[str, Any]], float],
    param_grid: dict[str, list],
    budget: int,
) -> tuple[dict[str, Any], float]:
    """Evaluate candidate configs in deterministic (itertools) order, at most
    ``budget`` of them; return (best_params, best_score)."""
    if budget < 1:
        raise PipelineError("hpo budget must be >= 1")
    keys = sorted(param_grid)
    combos = list(itertools.product(*(param_grid[k] for k in keys)))[:budget]
    best_params, best_score = None, float("-inf")
    for values in combos:
        params = dict(zip(keys, values))
        score = evaluate_config(params)
        if score > best_score:
            best_params, best_score = params, score
    if best_params is None:  # pragma: no cover - combos nonempty when budget >= 1
        raise PipelineError("no candidate evaluated")
    return best_params, best_score
