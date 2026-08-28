"""
examples/01_xor.py — Learning the XOR function with a 2-layer MLP.

XOR truth table:
    0 ^ 0 = 0
    0 ^ 1 = 1
    1 ^ 0 = 1
    1 ^ 1 = 0

A linear model cannot separate this (it's not linearly separable), so we
need at least one hidden layer. This is the classic demo of why depth matters.

Run:
    python examples/01_xor.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import neural_network_framework.nn as nn
from neural_network_framework.tensor import Tensor
from neural_network_framework.optim import Adam
from neural_network_framework.loss import BCEWithLogitsLoss


def main():
    np.random.seed(42)

    # XOR dataset — all 4 input combinations
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float32))

    # 2-layer MLP: 2 → 8 → 1
    model = nn.Sequential(
        nn.Linear(2, 8, init="xavier"),
        nn.Tanh(),
        nn.Linear(8, 1, init="xavier"),
    )

    optimizer = Adam(model.parameters(), lr=0.05)
    criterion = BCEWithLogitsLoss()

    print("=" * 50)
    print("  XOR Problem — Training with Adam")
    print("=" * 50)
    print(f"{'Step':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print("-" * 32)

    for step in range(1, 2001):
        # Forward pass
        logits = model(X)
        loss = criterion(logits, y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        if step % 200 == 0 or step == 1:
            # Accuracy
            preds = (logits.data > 0).astype(int)
            acc = (preds == y.data.astype(int)).mean()
            print(f"{step:>6}  {loss.item():>10.6f}  {acc * 100:>9.1f}%")

    print("-" * 32)
    print("\nFinal predictions (sigmoid applied):")
    sigmoid_out = 1.0 / (1.0 + np.exp(-logits.data))
    for (x0, x1), prob, label in zip(X.data, sigmoid_out.flatten(), y.data.flatten()):
        predicted = int(prob > 0.5)
        correct = "✓" if predicted == int(label) else "✗"
        print(f"  {int(x0)} XOR {int(x1)} = {int(label)}  →  p={prob:.4f}  [{correct}]")


if __name__ == "__main__":
    main()
