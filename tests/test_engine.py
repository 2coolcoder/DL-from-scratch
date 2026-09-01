"""Gradient-correctness tests: our engine vs torch.autograd and vs finite differences."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlscratch.engine import Value, dot, topological_order
from dlscratch.losses import softmax_cross_entropy, tree_sum
from dlscratch.nn import MLP

TOL = 1e-9


def test_mixed_expression_matches_torch():
    import torch

    def build(V, ops):
        a, b, c = V
        e = a * b + c
        f = (e * 2 - a) ** 3
        g = ops["tanh"](f * 0.01) + ops["exp"](b * 0.5)
        h = g / (c * c + 1.0) + ops["log"](a * a + 2.0)
        return h * h - c

    a, b, c = Value(2.0), Value(-3.0), Value(4.5)
    ours = build(
        (a, b, c),
        {"tanh": lambda v: v.tanh(), "exp": lambda v: v.exp(), "log": lambda v: v.log()},
    )
    ours.backward()

    ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(4.5, dtype=torch.float64, requires_grad=True)
    theirs = build((ta, tb, tc), {"tanh": torch.tanh, "exp": torch.exp, "log": torch.log})
    theirs.backward()

    assert abs(ours.data - theirs.item()) < TOL
    for mine, ref in ((a, ta), (b, tb), (c, tc)):
        assert abs(mine.grad - ref.grad.item()) < TOL, (mine.grad, ref.grad.item())


def test_relu_and_rops_match_torch():
    import torch

    for x0 in (-1.7, 0.3, 2.9):
        x = Value(x0)
        out = (3 - x).relu() * (2 / (x * x + 4.0)) + (-x)
        out.backward()

        tx = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        tout = torch.relu(3 - tx) * (2 / (tx * tx + 4.0)) + (-tx)
        tout.backward()

        assert abs(out.data - tout.item()) < TOL
        assert abs(x.grad - tx.grad.item()) < TOL


def test_fused_dot_matches_unfused_and_torch():
    import torch

    rng = random.Random(0)
    n = 12
    xs_v = [rng.uniform(-2, 2) for _ in range(n)]
    ws_v = [rng.uniform(-2, 2) for _ in range(n)]
    b_v = rng.uniform(-1, 1)

    # fused
    xs = [Value(v) for v in xs_v]
    ws = [Value(v) for v in ws_v]
    b = Value(b_v)
    fused = dot(xs, ws, b).tanh()
    fused.backward()

    # unfused: the same maths written with __mul__ / __add__
    xs2 = [Value(v) for v in xs_v]
    ws2 = [Value(v) for v in ws_v]
    b2 = Value(b_v)
    acc = b2
    for x, w in zip(xs2, ws2):
        acc = acc + x * w
    unfused = acc.tanh()
    unfused.backward()

    tx = torch.tensor(xs_v, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(ws_v, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_v, dtype=torch.float64, requires_grad=True)
    tout = torch.tanh(tx @ tw + tb)
    tout.backward()

    assert abs(fused.data - unfused.data) < TOL
    assert abs(fused.data - tout.item()) < TOL
    for i in range(n):
        assert abs(xs[i].grad - xs2[i].grad) < TOL
        assert abs(ws[i].grad - ws2[i].grad) < TOL
        assert abs(xs[i].grad - tx.grad[i].item()) < TOL
        assert abs(ws[i].grad - tw.grad[i].item()) < TOL
    assert abs(b.grad - tb.grad.item()) < TOL


def test_mlp_backward_matches_torch():
    import torch

    dims = [6, 5, 4, 3]
    model = MLP(dims, hidden_act="tanh", seed=7)
    x = [0.4, -1.1, 0.9, 0.05, -0.3, 1.6]
    target = 2

    loss = softmax_cross_entropy(model(x), target)
    loss.backward()

    tparams, tlayers = [], []
    for layer in model.layers:
        W = torch.tensor(
            [[w.data for w in n.weights] for n in layer.neurons],
            dtype=torch.float64,
            requires_grad=True,
        )
        bvec = torch.tensor(
            [n.bias.data for n in layer.neurons], dtype=torch.float64, requires_grad=True
        )
        tlayers.append((W, bvec))
        tparams += [W, bvec]

    h = torch.tensor(x, dtype=torch.float64)
    for i, (W, bvec) in enumerate(tlayers):
        h = W @ h + bvec
        if i < len(tlayers) - 1:
            h = torch.tanh(h)
    tloss = torch.nn.functional.cross_entropy(h.unsqueeze(0), torch.tensor([target]))
    tloss.backward()

    assert abs(loss.data - tloss.item()) < 1e-9
    for layer, (W, bvec) in zip(model.layers, tlayers):
        for j, neuron in enumerate(layer.neurons):
            for i, w in enumerate(neuron.weights):
                assert abs(w.grad - W.grad[j, i].item()) < TOL, (w.grad, W.grad[j, i])
            assert abs(neuron.bias.grad - bvec.grad[j].item()) < TOL


def test_finite_differences():
    """Every unary op, checked numerically -- no torch involved."""
    ops = {
        "tanh": (lambda v: v.tanh(), 0.7),
        "relu": (lambda v: v.relu(), 0.7),
        "exp": (lambda v: v.exp(), 0.4),
        "log": (lambda v: v.log(), 1.8),
        "pow3": (lambda v: v ** 3, -1.3),
        "recip": (lambda v: 1.0 / v, 2.2),
    }
    eps = 1e-6
    for name, (fn, x0) in ops.items():
        x = Value(x0)
        y = fn(x)
        y.backward()
        numeric = (fn(Value(x0 + eps)).data - fn(Value(x0 - eps)).data) / (2 * eps)
        assert abs(x.grad - numeric) < 1e-5, (name, x.grad, numeric)


def test_diamond_accumulates_once():
    """A node reused on two paths must receive the sum of both, not a double
    visit -- this is what the original list-based topo sort got right but slowly."""
    a = Value(3.0)
    b = a * a + a  # a appears three times in the graph
    b.backward()
    assert abs(b.data - 12.0) < TOL
    assert abs(a.grad - 7.0) < TOL  # 2a + 1


def test_backward_is_not_recursive():
    """A 50k-deep chain would blow the recursion limit under the old engine."""
    v = Value(0.001)
    out = v
    for _ in range(50_000):
        out = out + v
    out.backward()
    assert abs(out.data - 0.001 * 50_001) < 1e-6
    assert abs(v.grad - 50_001) < 1e-6
    assert len(topological_order(out)) == 50_001


def test_backward_zeroes_by_default():
    """The notebook's zero_grad() silently no-op'd, so grads accumulated across
    steps.  Repeated backward() must be idempotent unless asked otherwise."""
    a = Value(2.0)
    out = a * 3
    out.backward()
    first = a.grad
    out.backward()
    assert a.grad == first
    out.backward(accumulate=True)
    assert abs(a.grad - 2 * first) < TOL


def test_tree_sum_matches_plain_sum():
    vals = [Value(float(i)) for i in range(37)]
    total = tree_sum(vals)
    total.backward()
    assert abs(total.data - sum(range(37))) < TOL
    assert all(abs(v.grad - 1.0) < TOL for v in vals)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
