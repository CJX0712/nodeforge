"""Metrics. Convention: higher is better for every metric produced here."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score as _sk_acc
from sklearn.metrics import f1_score as _sk_f1

from nodeforge.core.errors import EvalError


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise EvalError("length mismatch between y_true and y_pred")
    return float(_sk_acc(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise EvalError("length mismatch between y_true and y_pred")
    return float(_sk_f1(y_true, y_pred, average="macro"))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"acc": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}
