"""
viz.py — Computation graph visualizer for neural_network_framework.

Renders the backward graph of any Tensor as either:
  • ASCII art  (terminal, zero extra dependencies)
  • Graphviz DOT source (export to SVG/PNG with the system `dot` command)

Technical novelty
-----------------
Unlike micrograd's scalar-only visualizer, this renderer works correctly
for *tensor* operations — it shows per-node shapes, gradient norms after
backward(), op categories, and parameter counts.  Shared nodes in DAG
branches are identified by stable IDs and never duplicated.

Usage
-----
    from neural_network_framework.viz import draw_graph

    loss = model(x)
    draw_graph(loss)                            # ASCII to stdout (before backward)
    loss.backward()
    draw_graph(loss, show_grads=True)           # annotated with ‖∇‖ per node
    draw_graph(loss, fmt="dot", path="graph")   # saves graph.dot (and graph.svg
                                                # if graphviz is installed)

    # Inspect a sub-graph (not just the loss):
    hidden = model.layers[0](x)
    draw_graph(hidden)
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from neural_network_framework.tensor import Tensor


# ---------------------------------------------------------------------------
# Op → category mapping
# ---------------------------------------------------------------------------

_OP_CATEGORY: Dict[str, str] = {
    # Arithmetic
    "Add": "arith",  "Sub": "arith",  "Mul": "arith",
    "Div": "arith",  "Pow": "arith",
    # Linear algebra
    "MatMul": "linalg",  "Transpose": "linalg",
    "Reshape": "linalg", "Flatten": "linalg",
    # Reductions
    "Sum": "reduce",  "Mean": "reduce",  "Max": "reduce",
    # Element-wise math
    "Exp": "elemwise",  "Log": "elemwise",
    "Abs": "elemwise",  "Sqrt": "elemwise",
    # Activations
    "ReLU": "activation",      "LeakyReLU": "activation",
    "Sigmoid": "activation",   "Tanh": "activation",
    "GELU": "activation",      "Softmax": "activation",
    "LogSoftmax": "activation",
    # Layers
    "Dropout": "layer",  "BatchNorm1d": "layer",  "Embedding": "layer",
    # Losses
    "MSELoss": "loss",           "BCELoss": "loss",
    "BCEWithLogitsLoss": "loss", "CrossEntropyLoss": "loss",
    "NLLLoss": "loss",
}

# Symbol shown in ASCII boxes per category
_CAT_SYMBOL: Dict[str, str] = {
    "arith":      "⊕",
    "linalg":     "⊗",
    "reduce":     "Σ",
    "elemwise":   "ƒ",
    "activation": "σ",
    "layer":      "◈",
    "loss":       "ℒ",
    "leaf":       "✦",
    "unknown":    "•",
}

# Graphviz fill-colours per category
_DOT_COLOR: Dict[str, str] = {
    "arith":      "#AED6F1",   # light blue
    "linalg":     "#A9DFBF",   # light green
    "reduce":     "#F9E79F",   # light yellow
    "elemwise":   "#D2B4DE",   # light purple
    "activation": "#85C1E9",   # blue
    "layer":      "#FAD7A0",   # orange
    "loss":       "#F1948A",   # red-pink
    "leaf_param": "#F7DC6F",   # gold
    "leaf_data":  "#D5D8DC",   # grey
    "unknown":    "#FFFFFF",
}


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def _collect(root: Tensor) -> Tuple[List[Tensor], List[Tuple[int, int]]]:
    """
    BFS from *root* through all ``_prev`` links.

    Returns
    -------
    nodes : list[Tensor]
        Every reachable tensor, BFS order (root first).
    edges : list[(producer_id, consumer_id)]
        Forward-direction edges: the producer's Python ``id`` → the
        consumer's Python ``id``.  Because ``node._prev`` stores the
        *inputs* to ``node``, each element of ``_prev`` is a producer.
    """
    visited: Set[int] = set()
    nodes: List[Tensor] = []
    edges: List[Tuple[int, int]] = []

    queue: deque[Tensor] = deque([root])
    while queue:
        node = queue.popleft()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        nodes.append(node)
        for parent in node._prev:
            pid = id(parent)
            edges.append((pid, nid))      # parent produced → node consumed
            if pid not in visited:
                queue.append(parent)

    return nodes, edges


def _topo_sort(nodes: List[Tensor], edges: List[Tuple[int, int]]) -> List[Tensor]:
    """
    Kahn's algorithm: returns nodes in topological order,
    leaves (no producers) first, root last.
    """
    id_to_node: Dict[int, Tensor] = {id(n): n for n in nodes}
    in_degree: Dict[int, int] = {id(n): 0 for n in nodes}
    adj: Dict[int, List[int]] = defaultdict(list)

    for src, dst in edges:
        if src in in_degree and dst in in_degree:
            in_degree[dst] += 1
            adj[src].append(dst)

    queue: deque[int] = deque(
        nid for nid, deg in in_degree.items() if deg == 0
    )
    result: List[Tensor] = []
    while queue:
        nid = queue.popleft()
        result.append(id_to_node[nid])
        for nbr in adj[nid]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    return result   # leaves → root


def _depth_from_root(root: Tensor, nodes: List[Tensor]) -> Dict[int, int]:
    """
    BFS from root following ``_prev`` backward.
    Root gets depth 0; leaves get the largest depth values.
    """
    depths: Dict[int, int] = {id(root): 0}
    queue: deque[Tensor] = deque([root])
    visited: Set[int] = set()

    while queue:
        node = queue.popleft()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        for parent in node._prev:
            pid = id(parent)
            new_d = depths[nid] + 1
            if pid not in depths or depths[pid] < new_d:
                depths[pid] = new_d
            if pid not in visited:
                queue.append(parent)

    # Nodes unreachable from root (shouldn't happen) get depth = max + 1
    max_d = max(depths.values(), default=0)
    for n in nodes:
        if id(n) not in depths:
            depths[id(n)] = max_d + 1

    return depths


# ---------------------------------------------------------------------------
# Node property helpers
# ---------------------------------------------------------------------------

def _is_leaf(t: Tensor) -> bool:
    return len(t._prev) == 0


def _is_param(t: Tensor) -> bool:
    return _is_leaf(t) and t.requires_grad


def _category(t: Tensor) -> str:
    if _is_leaf(t):
        return "leaf"
    return _OP_CATEGORY.get(t._op, "unknown")


def _shape_str(t: Tensor) -> str:
    if t.data.ndim == 0 or t.data.size == 1:
        return "scalar"
    return "(" + "×".join(str(s) for s in t.shape) + ")"


def _grad_norm_str(t: Tensor) -> str:
    if t.grad is None:
        return ""
    gn = float(np.linalg.norm(t.grad))
    # Flag pathological gradients
    if np.isnan(gn):
        flag = " ⚠ NaN"
    elif np.isinf(gn):
        flag = " ⚠ Inf"
    elif gn < 1e-7 and t.requires_grad:
        flag = " ⚠ vanish"
    elif gn > 1e3:
        flag = " ⚠ explode"
    else:
        flag = ""
    return f"‖∇‖={gn:.4f}{flag}"


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------

def _render_ascii(
    root: Tensor,
    nodes: List[Tensor],
    edges: List[Tuple[int, int]],
    show_grads: bool,
) -> str:
    """
    Renders the graph as a vertical ASCII diagram.

    Layout strategy
    ~~~~~~~~~~~~~~~
    Nodes are sorted topologically (leaves at the top, root at the bottom).
    Each node gets a box.  Arrows below each box list the downstream
    consumers.  DAG merges (a node feeding multiple consumers) are handled
    by listing all consumer IDs on the arrow line.
    """
    topo = _topo_sort(nodes, edges)      # leaves first
    depths = _depth_from_root(root, nodes)
    max_depth = max(depths.values(), default=0)

    # Stable short IDs  (used to cross-reference shared nodes)
    node_sid: Dict[int, str] = {id(n): f"#{i:02d}" for i, n in enumerate(topo)}

    # Build forward adjacency: producer → list of consumers
    consumers: Dict[int, List[Tensor]] = defaultdict(list)
    node_by_id: Dict[int, Tensor] = {id(n): n for n in nodes}
    for src_id, dst_id in edges:
        if dst_id in node_by_id:
            consumers[src_id].append(node_by_id[dst_id])

    # Summary statistics
    n_params = sum(1 for n in nodes if _is_param(n))
    param_elems = sum(n.data.size for n in nodes if _is_param(n))
    root_val = (
        f"  │  output = {root.data.item():.6f}"
        if root.data.size == 1 else ""
    )

    W = 62
    lines: List[str] = []

    def bar(c="━"): lines.append(c * W)

    bar()
    lines.append(f"  neural_network_framework  ·  Computation Graph{root_val}")
    lines.append(
        f"  Nodes: {len(nodes)}   Edges: {len(edges)}   "
        f"Depth: {max_depth}   Params: {n_params} ({param_elems:,} elements)"
    )
    bar()
    lines.append("  Direction of forward pass: TOP (inputs) → BOTTOM (output)")
    bar("─")
    lines.append("")

    shown: Set[int] = set()

    for node in topo:
        nid = id(node)
        sid = node_sid[nid]
        cat = _category(node)
        sym = _CAT_SYMBOL.get(cat, "•")

        if _is_leaf(node):
            kind = "PARAM" if _is_param(node) else "DATA "
            op_part = f"{sym} {kind}"
        else:
            op_part = f"{sym} {node._op or '?'}"

        shape_part = _shape_str(node)
        grad_part = f"   {_grad_norm_str(node)}" if show_grads else ""

        # Check if this node was already drawn (DAG merge — seen from another path)
        if nid in shown:
            lines.append(f"  {sid}  ↑ (already drawn above)")
        else:
            shown.add(nid)
            # Box
            inner = f" {sid}  {op_part:<18}  {shape_part:<14}{grad_part} "
            w = len(inner)
            lines.append("  ┌" + "─" * w + "┐")
            lines.append("  │" + inner    + "│")
            lines.append("  └" + "─" * w + "┘")

        # Arrow to consumer(s)
        cons = consumers.get(nid, [])
        if cons:
            if len(cons) == 1:
                lines.append("       │")
                lines.append("       ▼")
            else:
                c_ids = ", ".join(node_sid[id(c)] for c in cons)
                lines.append(f"       │ ──► feeds {c_ids}")
                lines.append("       ▼")
        else:
            # No consumers → this is the root output
            pass

    lines.append("")
    bar("─")
    # Legend
    lines.append("  Legend:")
    cats_used = {_category(n) for n in nodes}
    for cat in ["loss", "linalg", "activation", "arith", "reduce",
                "elemwise", "layer", "leaf", "unknown"]:
        if cat in cats_used:
            lines.append(f"    {_CAT_SYMBOL[cat]}  {cat}")
    if show_grads:
        lines.append("  ⚠  gradient health flags: vanish (<1e-7)  explode (>1e3)  NaN/Inf")
    bar()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz DOT renderer
# ---------------------------------------------------------------------------

def _render_dot(
    nodes: List[Tensor],
    edges: List[Tuple[int, int]],
    show_grads: bool,
) -> str:
    """
    Render the graph as Graphviz DOT source.

    Nodes
    -----
    • Op nodes   → rounded rectangles, colour-coded by category
    • Param leaves → gold diamonds
    • Data leaves  → grey rectangles

    Edges
    -----
    Forward direction (producer → consumer), labelled with the producer
    tensor's shape.
    """
    dot_id: Dict[int, str] = {id(n): f"n{i}" for i, n in enumerate(nodes)}

    lines: List[str] = [
        "digraph neural_network_frameworkComputationGraph {",
        '  graph [rankdir=TB, splines=ortho, bgcolor="#F8F9FA", '
        '         fontname="Helvetica Neue", label="neural_network_framework computation graph", '
        '         labelloc=t, fontsize=14];',
        '  node  [fontname="Helvetica Neue", fontsize=10, '
        '         style="filled,rounded", margin="0.25,0.12"];',
        '  edge  [color="#555555", arrowsize=0.75, fontsize=8, '
        '         fontcolor="#666666", fontname="Helvetica Neue"];',
        "",
    ]

    for node in nodes:
        nid = dot_id[id(node)]
        cat = _category(node)

        # Label lines
        if _is_leaf(node):
            kind = "PARAM" if _is_param(node) else "DATA"
            label_lines = [kind, _shape_str(node)]
            if show_grads:
                g = _grad_norm_str(node)
                if g:
                    label_lines.append(g)
            color = _DOT_COLOR["leaf_param" if _is_param(node) else "leaf_data"]
            shape = "diamond" if _is_param(node) else "box"
            penwidth = "2.0" if _is_param(node) else "1.0"
        else:
            sym = _CAT_SYMBOL.get(cat, "•")
            label_lines = [f"{sym}  {node._op or '?'}", _shape_str(node)]
            if show_grads:
                g = _grad_norm_str(node)
                if g:
                    label_lines.append(g)
            color = _DOT_COLOR.get(cat, _DOT_COLOR["unknown"])
            shape = "box"
            penwidth = "1.0"

        label = "\\n".join(label_lines)
        lines.append(
            f'  {nid} [label="{label}", shape={shape}, '
            f'fillcolor="{color}", penwidth={penwidth}, color="#333333"];'
        )

    lines.append("")

    # Edges (forward direction: producer → consumer)
    node_by_id: Dict[int, Tensor] = {id(n): n for n in nodes}
    for src_id, dst_id in edges:
        src_dot = dot_id.get(src_id)
        dst_dot = dot_id.get(dst_id)
        if src_dot is None or dst_dot is None:
            continue
        src_node = node_by_id.get(src_id)
        edge_label = _shape_str(src_node) if src_node else ""
        lines.append(f'  {src_dot} -> {dst_dot} [label="{edge_label}"];')

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_graph(
    tensor: "Tensor",
    fmt: str = "ascii",
    show_grads: bool = True,
    path: Optional[str] = None,
    render_svg: bool = False,
) -> str:
    """
    Render the computation graph of a Tensor.

    Parameters
    ----------
    tensor : Tensor
        The root tensor to visualize (typically the loss).  The graph is
        traced backward through its ``_prev`` links to all reachable nodes.
    fmt : {"ascii", "dot"}
        Output format.
        - ``"ascii"`` (default) — human-readable terminal output with
          box-drawing characters.
        - ``"dot"`` — Graphviz DOT source, renderable with ``dot -Tsvg``.
    show_grads : bool
        If True (default), annotate each node with its gradient norm
        ``‖∇‖``.  Only meaningful *after* ``tensor.backward()`` has been
        called; before backward, all norms are absent.
        Automatically adds health flags (vanishing / exploding / NaN).
    path : str or None
        If given, write the output to ``<path>.txt`` (ASCII) or
        ``<path>.dot`` (DOT) instead of printing to stdout.
    render_svg : bool
        If True and ``path`` is given and ``fmt="dot"``, attempt to run
        ``dot -Tsvg <path>.dot -o <path>.svg`` to produce an SVG file.
        Silently skips if Graphviz is not installed.

    Returns
    -------
    str
        The rendered graph string (also printed/saved based on arguments).

    Examples
    --------
    >>> from neural_network_framework.viz import draw_graph
    >>> import neural_network_framework as ng
    >>> from neural_network_framework.nn.layers import Linear
    >>> from neural_network_framework.nn.activations import ReLU
    >>> from neural_network_framework.loss import MSELoss
    >>>
    >>> model = ng.Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    >>> x = ng.Tensor([[1.0, 0.0]], requires_grad=False)
    >>> y = ng.Tensor([[1.0]])
    >>> loss = MSELoss()(model(x), y)
    >>> draw_graph(loss)                       # ASCII before backward
    >>> loss.backward()
    >>> draw_graph(loss, show_grads=True)      # annotated with grad norms
    >>> draw_graph(loss, fmt="dot", path="xor_graph", render_svg=True)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(
            f"draw_graph() expects a neural_network_framework Tensor, got {type(tensor).__name__!r}"
        )

    nodes, edges = _collect(tensor)

    if fmt == "dot":
        output = _render_dot(nodes, edges, show_grads)
        ext = ".dot"
    elif fmt == "ascii":
        output = _render_ascii(tensor, nodes, edges, show_grads)
        ext = ".txt"
    else:
        raise ValueError(f"fmt must be 'ascii' or 'dot', got {fmt!r}")

    if path is not None:
        full_path = path + ext
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"[neural_network_framework.viz] Saved to '{full_path}'  ({len(nodes)} nodes)")

        if render_svg and fmt == "dot":
            svg_path = path + ".svg"
            ret = os.system(f'dot -Tsvg "{full_path}" -o "{svg_path}"')
            if ret == 0:
                print(f"[neural_network_framework.viz] SVG rendered → '{svg_path}'")
            else:
                print(
                    "[neural_network_framework.viz] 'dot' not found — "
                    "install Graphviz (https://graphviz.org) to auto-render SVG"
                )
    else:
        print(output)

    return output
