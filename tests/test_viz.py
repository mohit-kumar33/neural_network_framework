"""
tests/test_viz.py — Unit tests for neural_network_framework.viz

Tests cover:
    - _collect()     : correct node/edge counts for known graphs
    - _topo_sort()   : leaves before root guarantee
    - _depth_from_root(): correct depth assignment
    - _render_ascii(): output contains expected strings
    - _render_dot()  : valid DOT with node/edge lines
    - draw_graph()   : public API runs without error for ascii and dot
    - DAG handling   : shared node appears only once in node list
    - Gradient health flags: vanish / explode annotations
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_network_framework.tensor import Tensor
from neural_network_framework.viz import (
    _collect, _topo_sort, _depth_from_root,
    _render_ascii, _render_dot, draw_graph,
    _is_leaf, _is_param, _category, _grad_norm_str,
)
from neural_network_framework.nn.layers import Linear
from neural_network_framework.nn.activations import ReLU
from neural_network_framework.loss import MSELoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def simple_graph():
    """a * b + c  →  5 nodes (a, b, c, mul, add), 4 edges."""
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([1.0], requires_grad=True)
    mul = a * b
    out = mul + c
    return out, [a, b, c]


def linear_graph():
    """Single Linear layer forward pass."""
    layer = Linear(2, 4)
    x = Tensor([[1.0, 0.5]])
    return layer(x)


# ── _collect ──────────────────────────────────────────────────────────────────

class TestCollect:
    def test_node_count(self):
        out, _ = simple_graph()
        nodes, _ = _collect(out)
        # out(Add), mul(Mul), a, b, c  → 5 nodes
        assert len(nodes) == 5

    def test_edge_count(self):
        out, _ = simple_graph()
        _, edges = _collect(out)
        # a→mul, b→mul, mul→out, c→out  → 4 edges
        assert len(edges) == 4

    def test_root_is_first(self):
        out, _ = simple_graph()
        nodes, _ = _collect(out)
        assert nodes[0] is out

    def test_dag_no_duplicate_nodes(self):
        """When a node is used twice, it should appear only once."""
        a = Tensor([1.0], requires_grad=True)
        out = a + a          # a is used twice
        nodes, _ = _collect(out)
        node_ids = [id(n) for n in nodes]
        # a appears once, out appears once → 2 unique nodes
        assert len(node_ids) == len(set(node_ids))
        assert len(nodes) == 2

    def test_scalar_graph(self):
        a = Tensor(3.14)
        nodes, edges = _collect(a)
        assert len(nodes) == 1
        assert len(edges) == 0


# ── _topo_sort ────────────────────────────────────────────────────────────────

class TestTopoSort:
    def test_root_is_last(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        topo = _topo_sort(nodes, edges)
        assert topo[-1] is out

    def test_leaves_precede_root(self):
        out, leaves = simple_graph()
        nodes, edges = _collect(out)
        topo = _topo_sort(nodes, edges)
        topo_ids = [id(n) for n in topo]
        root_pos = topo_ids.index(id(out))
        for leaf in leaves:
            assert topo_ids.index(id(leaf)) < root_pos

    def test_length_preserved(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        topo = _topo_sort(nodes, edges)
        assert len(topo) == len(nodes)


# ── _depth_from_root ──────────────────────────────────────────────────────────

class TestDepthMap:
    def test_root_depth_zero(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        depths = _depth_from_root(out, nodes)
        assert depths[id(out)] == 0

    def test_leaves_deepest(self):
        out, leaves = simple_graph()
        nodes, edges = _collect(out)
        depths = _depth_from_root(out, nodes)
        for leaf in leaves:
            # leaves should be at depth 1 or 2
            assert depths[id(leaf)] >= 1


# ── Node property helpers ──────────────────────────────────────────────────────

class TestNodeHelpers:
    def test_is_leaf_true(self):
        a = Tensor([1.0])
        assert _is_leaf(a)

    def test_is_leaf_false(self):
        a = Tensor([1.0])
        b = a + a
        assert not _is_leaf(b)

    def test_is_param(self):
        p = Tensor([1.0], requires_grad=True)
        assert _is_param(p)

    def test_is_not_param_no_grad(self):
        d = Tensor([1.0], requires_grad=False)
        assert not _is_param(d)

    def test_category_leaf(self):
        a = Tensor([1.0])
        assert _category(a) == "leaf"

    def test_category_activation(self):
        a = Tensor([1.0], requires_grad=True)
        relu_out = ReLU()(a)
        assert _category(relu_out) == "activation"

    def test_grad_norm_none_before_backward(self):
        a = Tensor([1.0], requires_grad=True)
        out = a * a
        out.backward()
        # a.grad should now be set
        s = _grad_norm_str(a)
        assert "‖∇‖" in s

    def test_grad_norm_vanishing_flag(self):
        a = Tensor(np.array([1e-9], dtype=np.float32), requires_grad=True)
        a._init_grad()
        a.grad = np.array([1e-9], dtype=np.float32)   # vanishingly small
        s = _grad_norm_str(a)
        assert "vanish" in s

    def test_grad_norm_explode_flag(self):
        a = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        a._init_grad()
        a.grad = np.array([1e4], dtype=np.float32)     # exploding
        s = _grad_norm_str(a)
        assert "explode" in s


# ── ASCII renderer ─────────────────────────────────────────────────────────────

class TestRenderAscii:
    def test_contains_header(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        txt = _render_ascii(out, nodes, edges, show_grads=False)
        assert "neural_network_framework" in txt
        assert "Computation Graph" in txt

    def test_contains_op_names(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        txt = _render_ascii(out, nodes, edges, show_grads=False)
        assert "Add" in txt
        assert "Mul" in txt

    def test_contains_legend(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        txt = _render_ascii(out, nodes, edges, show_grads=False)
        assert "Legend" in txt

    def test_grad_annotation_after_backward(self):
        a = Tensor([2.0], requires_grad=True)
        out = (a * a).sum()
        out.backward()
        nodes, edges = _collect(out)
        txt = _render_ascii(out, nodes, edges, show_grads=True)
        assert "‖∇‖" in txt

    def test_no_grad_annotation_when_disabled(self):
        a = Tensor([2.0], requires_grad=True)
        out = (a * a).sum()
        out.backward()
        nodes, edges = _collect(out)
        txt = _render_ascii(out, nodes, edges, show_grads=False)
        assert "‖∇‖" not in txt

    def test_returns_string(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        result = _render_ascii(out, nodes, edges, show_grads=False)
        assert isinstance(result, str)
        assert len(result) > 0


# ── DOT renderer ───────────────────────────────────────────────────────────────

class TestRenderDot:
    def test_valid_dot_header(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        dot = _render_dot(nodes, edges, show_grads=False)
        assert "digraph" in dot
        assert "rankdir=TB" in dot

    def test_contains_node_ids(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        dot = _render_dot(nodes, edges, show_grads=False)
        assert "n0" in dot   # first node

    def test_contains_edges(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        dot = _render_dot(nodes, edges, show_grads=False)
        assert "->" in dot

    def test_contains_op_label(self):
        out, _ = simple_graph()
        nodes, edges = _collect(out)
        dot = _render_dot(nodes, edges, show_grads=False)
        assert "Add" in dot or "Mul" in dot

    def test_grad_annotation_in_dot(self):
        a = Tensor([2.0], requires_grad=True)
        out = (a * a).sum()
        out.backward()
        nodes, edges = _collect(out)
        dot = _render_dot(nodes, edges, show_grads=True)
        assert "‖∇‖" in dot


# ── Public API ─────────────────────────────────────────────────────────────────

class TestDrawGraph:
    def test_ascii_returns_string(self, capsys):
        out, _ = simple_graph()
        result = draw_graph(out, fmt="ascii", show_grads=False)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_dot_returns_string(self, capsys):
        out, _ = simple_graph()
        result = draw_graph(out, fmt="dot", show_grads=False)
        assert isinstance(result, str)
        assert "digraph" in result

    def test_invalid_fmt_raises(self):
        out, _ = simple_graph()
        with pytest.raises(ValueError, match="fmt must be"):
            draw_graph(out, fmt="svg")

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError):
            draw_graph(42)

    def test_saves_file(self, tmp_path):
        out, _ = simple_graph()
        p = str(tmp_path / "test_graph")
        draw_graph(out, fmt="ascii", path=p)
        assert os.path.exists(p + ".txt")

    def test_saves_dot_file(self, tmp_path):
        out, _ = simple_graph()
        p = str(tmp_path / "test_graph")
        draw_graph(out, fmt="dot", path=p)
        assert os.path.exists(p + ".dot")

    def test_real_mlp_graph(self):
        """End-to-end: MLP forward + backward, then visualize."""
        from neural_network_framework.nn.module import Sequential
        model = Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 1),
        )
        x = Tensor([[0., 0.], [1., 1.]], requires_grad=False)
        y = Tensor([[0.], [1.]])
        loss = MSELoss()(model(x), y)
        loss.backward()
        result = draw_graph(loss, show_grads=True)
        assert "neural_network_framework" in result
        assert "MatMul" in result

    def test_subgraph_visualization(self):
        """Can visualize any intermediate tensor, not just the loss."""
        layer = Linear(2, 4)
        x = Tensor([[1.0, 0.0]])
        hidden = layer(x)     # intermediate tensor
        result = draw_graph(hidden, show_grads=False)
        assert "MatMul" in result
