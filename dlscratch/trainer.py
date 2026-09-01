"""Minibatch SGD training loop over the scalar engine."""

import random
import time

from .engine import Value
from .losses import softmax_cross_entropy, tree_sum
from .metrics import accuracy, argmax, confusion_matrix
from .optim import SGD

__all__ = ["evaluate", "predict", "train"]


def predict(model, X):
    """Predicted class indices.  Builds a graph per sample but never calls
    backward, so the nodes are dropped as soon as the logits are read."""
    return [argmax([o.data for o in model(x)]) for x in X]


def evaluate(model, X, y):
    """Mean cross-entropy and accuracy over a split (forward only)."""
    losses, preds = [], []
    for xi, yi in zip(X, y):
        logits = model(xi)
        vals = [o.data for o in logits]
        preds.append(argmax(vals))
        losses.append(softmax_cross_entropy(logits, yi).data)
    return sum(losses) / len(losses), accuracy(preds, y)


def train(
    model,
    data,
    epochs=50,
    lr=0.1,
    batch_size=16,
    momentum=0.9,
    weight_decay=0.0,
    lr_decay=1.0,
    seed=0,
    verbose=True,
    log_every=1,
):
    """Train ``model`` on ``data`` and return a history dict.

    One graph is built per minibatch: the batch loss is the tree-sum of the
    per-sample cross-entropies, so a single backward pass covers the batch.
    """
    rng = random.Random(seed)
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    order = list(range(len(data.X_train)))

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [], "lr": [], "elapsed": [],
    }
    best = {"test_acc": -1.0, "epoch": -1, "params": None}
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        epoch_loss, seen = 0.0, 0

        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            terms = []
            for i in idx:
                logits = model(data.X_train[i])
                terms.append(softmax_cross_entropy(logits, data.y_train[i]))
            loss = tree_sum(terms) * (1.0 / len(terms))

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.data * len(idx)
            seen += len(idx)

        opt.lr *= lr_decay
        test_loss, test_acc = evaluate(model, data.X_test, data.y_test)
        train_acc = accuracy(predict(model, data.X_train), data.y_train)
        elapsed = time.perf_counter() - started

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss / seen)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(opt.lr)
        history["elapsed"].append(elapsed)

        if test_acc > best["test_acc"]:
            best = {
                "test_acc": test_acc,
                "epoch": epoch,
                "params": [p.data for p in model.parameters()],
            }

        if verbose and (epoch % log_every == 0 or epoch == epochs):
            print(
                f"  epoch {epoch:>3}/{epochs}  loss {epoch_loss / seen:.4f}  "
                f"train_acc {train_acc:.4f}  test_acc {test_acc:.4f}  ({elapsed:.1f}s)"
            )

    history["best_test_acc"] = best["test_acc"]
    history["best_epoch"] = best["epoch"]
    history["final_test_acc"] = history["test_acc"][-1]
    history["total_seconds"] = time.perf_counter() - started
    history["sec_per_epoch"] = history["total_seconds"] / epochs

    preds = predict(model, data.X_test)
    history["confusion_matrix"] = confusion_matrix(preds, data.y_test, data.n_classes)
    return history
