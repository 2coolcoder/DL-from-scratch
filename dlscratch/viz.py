"""Optional graphviz rendering of a computation graph.

``graphviz`` is not a hard dependency -- importing ``dlscratch`` must work
without it, so the import happens inside :func:`draw_dot`.
"""

from .engine import topological_order

__all__ = ["trace", "draw_dot"]


def trace(root):
    nodes = topological_order(root)
    edges = {(child, node) for node in nodes for child in node._prev}
    return set(nodes), edges


def draw_dot(root, rankdir="LR"):
    try:
        from graphviz import Digraph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "draw_dot needs the graphviz python package and the graphviz binaries:\n"
            "  pip install graphviz && sudo apt-get install graphviz"
        ) from exc

    dot = Digraph(format="svg", graph_attr={"rankdir": rankdir})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
