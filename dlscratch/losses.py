"""Loss functions over ``Value`` graphs."""

import math

from .engine import Value

__all__ = ["tree_sum", "mse", "softmax_cross_entropy", "softmax"]


def tree_sum(values):
    """Sum by pairwise reduction so the graph is O(log n) deep, not O(n).

    A left-fold ``sum()`` over a 512-sample batch produces a 512-long chain;
    halving keeps the DAG shallow and the topological sort cache-friendly.
    """
    if not values:
        return Value(0.0)
    items = list(values)
    while len(items) > 1:
        items = [
            items[i] + items[i + 1] if i + 1 < len(items) else items[i]
            for i in range(0, len(items), 2)
        ]
    return items[0]


def mse(preds, targets):
    """Mean squared error.  ``preds`` may be scalars or per-sample lists."""
    assert len(preds) == len(targets)
    if preds and isinstance(preds[0], (list, tuple)):
        terms = [
            (p - t) ** 2
            for row, trow in zip(preds, targets)
            for p, t in zip(row, trow)
        ]
    else:
        terms = [(p - t) ** 2 for p, t in zip(preds, targets)]
    return tree_sum(terms) * (1.0 / len(terms))


def softmax_cross_entropy(logits, target):
    """``-log p[target]`` computed as a stable log-sum-exp.

    The max is subtracted as a plain float constant (it shifts the values but
    not the derivative), so no gradient flows through it -- the same trick
    frameworks implement with a detach.
    """
    m = max(l.data for l in logits)
    shifted = [l - m for l in logits]
    logZ = tree_sum([s.exp() for s in shifted]).log()
    return logZ - shifted[target]


def softmax(logits):
    """Probabilities from raw logit *floats* (no graph, for reporting only)."""
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    z = sum(exps)
    return [e / z for e in exps]
