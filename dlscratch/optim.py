"""Optimizers operating on flat lists of ``Value`` parameters."""

__all__ = ["SGD"]


class SGD:
    def __init__(self, params, lr=0.05, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [0.0] * len(self.params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            if self.momentum:
                self._velocity[i] = self.momentum * self._velocity[i] + g
                g = self._velocity[i]
            p.data -= self.lr * g
