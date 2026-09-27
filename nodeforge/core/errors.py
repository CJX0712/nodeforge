"""Error taxonomy. Ranges: E100-E199 data, E200-E299 model, E300-E399 training,
E400-E499 evaluation, E500-E599 pipeline/config."""
from __future__ import annotations


class NodeForgeError(Exception):
    """Base error for all NodeForge failures. Subclass ranges carry a code."""

    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class DataError(NodeForgeError):
    """Graph/data generator failures (invalid params, LFR infeasible, ...)."""

    code = "E100"


class ModelError(NodeForgeError):
    """Model construction/forward failures (unknown name, bad shapes)."""

    code = "E200"


class TrainError(NodeForgeError):
    """Training loop failures (NaN loss, empty train mask)."""

    code = "E300"


class EvalError(NodeForgeError):
    """Evaluation failures (empty test mask, label mismatch)."""

    code = "E400"


class PipelineError(NodeForgeError):
    """Pipeline/config failures (invalid config, bad output path)."""

    code = "E500"
