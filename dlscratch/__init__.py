"""A scalar reverse-mode autograd engine and the neural nets built on it."""

from .engine import Value, dot, topological_order
from .nn import MLP, Layer, Module, Neuron
from .losses import mse, softmax, softmax_cross_entropy, tree_sum
from .optim import SGD
from .metrics import accuracy, argmax, confusion_matrix, format_confusion_matrix
from .data import DATASETS, Dataset, load
from .trainer import evaluate, predict, train
from .viz import draw_dot, trace  # graphviz itself is imported lazily inside draw_dot

__version__ = "0.1.0"

__all__ = [
    "Value", "dot", "topological_order",
    "Module", "Neuron", "Layer", "MLP",
    "mse", "softmax_cross_entropy", "softmax", "tree_sum",
    "SGD",
    "accuracy", "argmax", "confusion_matrix", "format_confusion_matrix",
    "load", "Dataset", "DATASETS",
    "train", "evaluate", "predict",
    "draw_dot", "trace",
]
