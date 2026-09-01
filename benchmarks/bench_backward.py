"""Measure what actually made the notebook's engine unusable.

Two independent problems, benchmarked separately:

1. topological sort -- the notebook tracked visited nodes in a *list*, making
   the sort O(n^2) in the number of nodes;
2. graph size -- the notebook's neuron built ``2 * fan_in`` nodes per unit,
   where the fused ``dot`` op builds one.

Run:  python benchmarks/bench_backward.py
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlscratch.engine import Value, dot, topological_order
from dlscratch.losses import softmax_cross_entropy, tree_sum
from dlscratch.nn import MLP


# --- the notebook's original traversal, verbatim in behaviour ---------------
def legacy_topological_order(root):
    """Notebook version: recursive, and ``not in`` against a list."""
    node_list = []

    def traverse(node):
        for n in node._prev:
            if n not in node_list:  # O(len(node_list)) identity scan
                traverse(n)
        node_list.append(node)

    traverse(root)
    node_list.reverse()
    return node_list


def legacy_neuron_forward(xs, ws, bias):
    """Notebook version: ``out = out + x*w`` -- two nodes per input."""
    out = bias
    for x, w in zip(xs, ws):
        out = out + x * w
    return out


def build_graph(dims, n_samples, fused, seed=0):
    """Forward pass of an MLP over ``n_samples``, returning the batch loss."""
    rng = random.Random(seed)
    layers = []
    for i in range(1, len(dims)):
        layers.append(
            [
                (
                    [Value(rng.uniform(-0.3, 0.3)) for _ in range(dims[i - 1])],
                    Value(0.0),
                )
                for _ in range(dims[i])
            ]
        )

    terms = []
    for _ in range(n_samples):
        h = [Value(rng.uniform(-1, 1)) for _ in range(dims[0])]
        for li, layer in enumerate(layers):
            pre = [
                dot(h, ws, b) if fused else legacy_neuron_forward(h, ws, b)
                for ws, b in layer
            ]
            h = pre if li == len(layers) - 1 else [p.tanh() for p in pre]
        terms.append(softmax_cross_entropy(h, rng.randrange(dims[-1])))
    return tree_sum(terms) * (1.0 / n_samples)


def time_it(fn, repeat=1):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench_headline(dims, n_samples):
    """The honest apples-to-apples number: one full training step, built and
    differentiated exactly the way the notebook did it, versus the way this
    repo does it."""
    print(f"\n[0] ONE TRAINING STEP, notebook config vs this repo -- MLP {dims}, batch {n_samples}")
    sys.setrecursionlimit(1_000_000)
    results = {}

    for label, fused, topo in (
        ("notebook (scalar ops + list topo sort)", False, legacy_topological_order),
        ("this repo (fused dot + set topo sort)", True, topological_order),
    ):
        t = time.perf_counter()
        root = build_graph(dims, n_samples, fused=fused)
        order = topo(root)
        if topo is legacy_topological_order:
            order = order  # already reversed: parents-first
        else:
            order = list(reversed(order))
        for node in order:
            node.grad = 0.0
        root.grad = 1.0
        for node in order:
            node._backward()
        results[label] = time.perf_counter() - t
        print(f"    {label:<40}: {results[label]:8.3f} s")

    slow, fast = results.values()
    print(f"    speedup: {slow / fast:.0f}x")
    return results


def bench_topo(dims, n_samples):
    print(f"\n[1] topological sort -- MLP {dims}, batch {n_samples}, fused graph")
    root = build_graph(dims, n_samples, fused=True)
    n = len(topological_order(root))
    print(f"    graph size: {n:,} nodes")

    new = time_it(lambda: topological_order(root), repeat=3)
    print(f"    set-based, iterative (this repo) : {new * 1e3:9.2f} ms")

    sys.setrecursionlimit(1_000_000)
    try:
        old = time_it(lambda: legacy_topological_order(root))
        print(f"    list-based, recursive (notebook): {old * 1e3:9.2f} ms")
        print(f"    speedup: {old / new:.0f}x")
    except RecursionError:
        print("    list-based, recursive (notebook): RecursionError")


def bench_graph_size(dims, n_samples):
    print(f"\n[2] graph size + full forward/backward -- MLP {dims}, batch {n_samples}")
    for label, fused in (("fused dot op (this repo)", True), ("scalar +/* (notebook) ", False)):
        t = time.perf_counter()
        root = build_graph(dims, n_samples, fused=fused)
        fwd = time.perf_counter() - t
        n = len(topological_order(root))
        t = time.perf_counter()
        root.backward()
        bwd = time.perf_counter() - t
        print(
            f"    {label}: {n:>9,} nodes   forward {fwd * 1e3:8.1f} ms   "
            f"backward {bwd * 1e3:8.1f} ms"
        )


def bench_scaling(dims):
    """The point that matters: the old cost was *quadratic* in graph size.

    Each doubling of the batch should cost 2x if the engine is linear.  The
    notebook's engine costs ~4x, which is why it looked hung on MNIST rather
    than merely slow.
    """
    print(f"\n[4] scaling with batch size -- MLP {dims}, one full training step")
    print(f"    {'batch':>6} {'nodes':>10} {'notebook(s)':>12} {'ours(s)':>9} "
          f"{'ratio':>7}  cost per 2x batch")
    sys.setrecursionlimit(1_000_000)
    prev = None
    for bs in (1, 2, 4, 8, 16, 32):
        root = build_graph(dims, bs, fused=False)
        n = len(topological_order(root))
        t = time.perf_counter()
        order = legacy_topological_order(root)
        for node in order:
            node.grad = 0.0
        root.grad = 1.0
        for node in order:
            node._backward()
        old = time.perf_counter() - t

        fused_root = build_graph(dims, bs, fused=True)
        t = time.perf_counter()
        fused_root.backward()
        new = time.perf_counter() - t

        growth = f"{old / prev:.2f}x" if prev else "-"
        print(f"    {bs:>6} {n:>10,} {old:>12.3f} {new:>9.4f} {old / new:>6.0f}x  {growth:>6}")
        prev = old
    print("    (2x = linear, 4x = quadratic)")


def bench_end_to_end():
    print("\n[3] one training step, digits-sized net [64, 32, 10], batch 16")
    model = MLP([64, 32, 10], seed=0)
    rng = random.Random(0)
    X = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(16)]
    y = [rng.randrange(10) for _ in range(16)]

    def step():
        loss = tree_sum([softmax_cross_entropy(model(x), t) for x, t in zip(X, y)])
        loss.backward()

    t = time_it(step, repeat=5)
    print(f"    forward + backward: {t * 1e3:.1f} ms  ->  ~{1 / t:.0f} steps/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[49, 20, 10])
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    print("=" * 72)
    print("Autograd engine benchmarks")
    print("=" * 72)
    bench_headline(args.dims, args.batch)
    bench_topo(args.dims, args.batch)
    bench_graph_size(args.dims, args.batch)
    bench_end_to_end()
    bench_scaling(args.dims)
    print()
