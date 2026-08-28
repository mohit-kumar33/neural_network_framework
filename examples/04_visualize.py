"""
04_visualize.py — Computation Graph Visualization Demo

Demonstrates neural_network_framework.viz.draw_graph() on a real trained network.

Steps
-----
1.  Build a small MLP for XOR classification
2.  Run one forward pass → draw ASCII graph (no grads yet)
3.  Run backward() → draw ASCII graph annotated with ‖∇‖ per node
4.  Export a Graphviz DOT file (graph.dot) for rendering to SVG

Run
---
    python examples/04_visualize.py

To render the DOT to SVG (requires graphviz installed):
    dot -Tsvg graph.dot -o graph.svg
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neural_network_framework as ng
from neural_network_framework.nn.module import Sequential
from neural_network_framework.nn.layers import Linear
from neural_network_framework.nn.activations import ReLU, Sigmoid
from neural_network_framework.loss import BCEWithLogitsLoss
from neural_network_framework.viz import draw_graph

# ── Dataset: XOR ─────────────────────────────────────────────────────────────
X = ng.Tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = ng.Tensor([[0.],      [1.],     [1.],     [0.]])

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential(
    Linear(2, 8, init="xavier"),
    ReLU(),
    Linear(8, 4, init="xavier"),
    ReLU(),
    Linear(4, 1, init="xavier"),
)

criterion = BCEWithLogitsLoss()

# ── Forward pass ──────────────────────────────────────────────────────────────
logits = model(X)
loss   = criterion(logits, y)

print("\n" + "=" * 62)
print("  STEP 1 — Graph BEFORE backward()  (no gradient info yet)")
print("=" * 62 + "\n")
draw_graph(loss, show_grads=False)

# ── Backward pass ─────────────────────────────────────────────────────────────
model.zero_grad()
loss.backward()

print("\n" + "=" * 62)
print("  STEP 2 — Graph AFTER backward()   (gradient norms visible)")
print("=" * 62 + "\n")
draw_graph(loss, show_grads=True)

# ── Export DOT ────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 3 — Exporting Graphviz DOT")
print("=" * 62 + "\n")

out_path = os.path.join(os.path.dirname(__file__), "..", "graph")
draw_graph(loss, fmt="dot", show_grads=True, path=out_path, render_svg=True)

print("\nDone!  Open graph.svg in a browser for an interactive view.")
print("If graphviz is not installed, run:")
print("   dot -Tsvg graph.dot -o graph.svg")
