"""Scalar reverse-mode autograd engine.

A ``Value`` is a single scalar that remembers the operation that produced it and
the operands it came from.  Calling :meth:`Value.backward` walks the resulting
DAG in reverse topological order and accumulates ``dself/dnode`` into every
node's ``.grad``.
"""

import math

__all__ = ["Value", "dot", "topological_order"]


def topological_order(root):
    """Return every node reachable from ``root``, parents before children.

    Iterative (no recursion limit) and O(V + E): ``visited`` is a set, so each
    node is expanded once and each edge inspected once.
    """
    topo, visited = [], set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            topo.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        for child in node._prev:
            if child not in visited:
                stack.append((child, False))
    return topo


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data, label="", parents=(), _op=None):
        self.data = data
        self.grad = 0.0
        self._prev = parents
        self._op = _op
        self.label = label
        self._backward = _noop

    # ---------------------------------------------------------------- ops ---
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, parents=(self, other), _op="+")

        def _backward():
            g = out.grad
            self.grad += g
            other.grad += g

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, parents=(self, other), _op="*")

        def _backward():
            g = out.grad
            self.grad += g * other.data
            other.grad += g * self.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, parents=(self,), _op=f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, parents=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, parents=(self,), _op="relu")

        def _backward():
            if out.data > 0:
                self.grad += out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, parents=(self,), _op="exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), parents=(self,), _op="log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    # ------------------------------------------------------ derived ops ---
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return (self ** -1) * other

    # ----------------------------------------------------------- engine ---
    def backward(self, accumulate=False):
        """Seed ``d self/d self = 1`` and propagate gradients through the DAG.

        By default every reachable node is zeroed first, so repeated calls do
        not silently sum gradients from previous steps.  Pass
        ``accumulate=True`` to add into whatever is already in ``.grad``.
        """
        topo = topological_order(self)
        if not accumulate:
            for node in topo:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Zero ``.grad`` on this node and everything it depends on."""
        for node in topological_order(self):
            node.grad = 0.0

    def __repr__(self):
        label = f", label={self.label}" if self.label else ""
        return f"Value(data={self.data}, grad={self.grad}{label})"


def _noop():
    return None


def dot(xs, ws, bias):
    """Fused affine unit: ``sum(x_i * w_i) + bias`` as a single graph node.

    Mathematically identical to chaining ``__mul__``/``__add__``, but builds one
    node with ``2*len(xs)+1`` parents instead of ``2*len(xs)`` intermediate
    nodes, which is what makes the engine fast enough for real datasets.
    """
    assert len(xs) == len(ws), f"length mismatch: {len(xs)} inputs vs {len(ws)} weights"
    total = bias.data
    for x, w in zip(xs, ws):
        total += x.data * w.data
    out = Value(total, parents=(*xs, *ws, bias), _op="dot")

    def _backward():
        g = out.grad
        for x, w in zip(xs, ws):
            x.grad += w.data * g
            w.grad += x.data * g
        bias.grad += g

    out._backward = _backward
    return out
