"""Neurons, layers and MLPs built on top of the scalar ``Value`` engine."""

import math
import random

from .engine import Value, dot

__all__ = ["Module", "Neuron", "Layer", "MLP"]

ACTIVATIONS = {
    "tanh": lambda v: v.tanh(),
    "relu": lambda v: v.relu(),
    None: lambda v: v,
}


def _init_scale(fan_in, fan_out, act):
    """Xavier/Glorot for tanh, He for relu -- keeps pre-activations off the
    saturated tails of tanh, which uniform(-1, 1) does not for wide layers."""
    if act == "relu":
        return math.sqrt(2.0 / fan_in)
    return math.sqrt(6.0 / (fan_in + fan_out))


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, n_in, act="tanh", fan_out=1, rng=random, label=""):
        scale = _init_scale(n_in, fan_out, act)
        self.weights = [
            Value(rng.uniform(-scale, scale), label=f"{label}w{i}") for i in range(n_in)
        ]
        self.bias = Value(0.0, label=f"{label}b")
        self.act = ACTIVATIONS[act]
        self.act_name = act

    def __call__(self, x):
        return self.act(dot(x, self.weights, self.bias))

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f"Neuron(in_dim={len(self.weights)}, act={self.act_name})"


class Layer(Module):
    def __init__(self, in_dim, out_dim, act="tanh", rng=random, label=""):
        self.neurons = [
            Neuron(in_dim, act=act, fan_out=out_dim, rng=rng, label=f"{label}n{i}_")
            for i in range(out_dim)
        ]
        self.in_dim, self.out_dim, self.act_name = in_dim, out_dim, act

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(in_dim={self.in_dim}, out_dim={self.out_dim}, act={self.act_name})"


class MLP(Module):
    """Fully connected network.

    ``dims`` is ``[n_in, h1, ..., n_out]``.  Hidden layers use ``hidden_act``;
    the output layer uses ``out_act`` (``None`` -> raw logits, which is what
    ``losses.softmax_cross_entropy`` expects).
    """

    def __init__(self, dims, hidden_act="tanh", out_act=None, seed=None):
        rng = random.Random(seed) if seed is not None else random
        self.dims = list(dims)
        self.layers = []
        for i in range(1, len(dims)):
            last = i == len(dims) - 1
            self.layers.append(
                Layer(
                    dims[i - 1],
                    dims[i],
                    act=out_act if last else hidden_act,
                    rng=rng,
                    label=f"l{i - 1}_",
                )
            )

    def __call__(self, x):
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def __repr__(self):
        return f"MLP(dims={self.dims}, params={len(self.parameters())})\n  " + "\n  ".join(
            repr(l) for l in self.layers
        )
