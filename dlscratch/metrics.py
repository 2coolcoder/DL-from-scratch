"""Evaluation helpers.  These consume plain floats, not ``Value`` graphs."""

__all__ = ["argmax", "accuracy", "confusion_matrix", "format_confusion_matrix"]


def argmax(seq):
    return max(range(len(seq)), key=seq.__getitem__)


def accuracy(pred_labels, true_labels):
    assert len(pred_labels) == len(true_labels)
    if not pred_labels:
        return 0.0
    hits = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    return hits / len(pred_labels)


def confusion_matrix(pred_labels, true_labels, n_classes):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for p, t in zip(pred_labels, true_labels):
        cm[t][p] += 1
    return cm


def format_confusion_matrix(cm, names=None):
    n = len(cm)
    names = names or [str(i) for i in range(n)]
    w = max(4, max(len(x) for x in names) + 1)
    head = " " * (w + 2) + "".join(f"{x:>{w}}" for x in names)
    rows = [f"{names[i]:>{w}} |" + "".join(f"{v:>{w}}" for v in cm[i]) for i in range(n)]
    return "\n".join([head] + rows)
