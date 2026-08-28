"""
examples/03_regression.py — Fitting a sine wave with a neural network.

Demonstrates:
- Regression with MSELoss
- Training loop with loss tracking
- Optional matplotlib visualisation

Run:
    python examples/03_regression.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import neural_network_framework.nn as nn
from neural_network_framework.tensor import Tensor
from neural_network_framework.optim import Adam
from neural_network_framework.loss import MSELoss


def generate_data(n=200, noise=0.05):
    """Generate noisy sine wave samples in [-π, π]."""
    np.random.seed(0)
    X = np.random.uniform(-np.pi, np.pi, n).astype(np.float32).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(n, 1).astype(np.float32)
    return X, y


def main():
    X_np, y_np = generate_data(n=300)
    X = Tensor(X_np)
    y = Tensor(y_np)

    # MLP: 1 → 64 → 64 → 1  with tanh activations
    model = nn.Sequential(
        nn.Linear(1, 64, init="xavier"),
        nn.Tanh(),
        nn.Linear(64, 64, init="xavier"),
        nn.Tanh(),
        nn.Linear(64, 1, init="xavier"),
    )

    optimizer = Adam(model.parameters(), lr=3e-3)
    criterion = MSELoss()

    print("=" * 50)
    print("  Sine-Wave Regression — Training")
    print("=" * 50)
    print(f"{'Epoch':>6}  {'MSE Loss':>12}")
    print("-" * 22)

    losses = []
    for epoch in range(1, 3001):
        pred = model(X)
        loss = criterion(pred, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 500 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>12.6f}")

    print("-" * 22)
    print(f"\nFinal MSE: {losses[-1]:.6f}")

    # Try to plot
    try:
        import matplotlib.pyplot as plt

        model.eval()
        X_test = np.linspace(-np.pi, np.pi, 200).astype(np.float32).reshape(-1, 1)
        y_true = np.sin(X_test)

        pred_test = model(Tensor(X_test))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Regression fit
        axes[0].scatter(X_np, y_np, s=10, alpha=0.5, label="Data", color="#64b5f6")
        axes[0].plot(X_test, y_true, "k--", linewidth=1.5, label="sin(x)", alpha=0.7)
        axes[0].plot(X_test, pred_test.data, color="#ef5350", linewidth=2, label="MLP fit")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Sine-Wave Regression")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Training loss curve
        axes[1].semilogy(losses, color="#26a69a", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss (log scale)")
        axes[1].set_title("Training Loss")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("examples/sine_regression.png", dpi=120, bbox_inches="tight")
        print("\nPlot saved to examples/sine_regression.png")
        plt.show()

    except ImportError:
        print("\n(Install matplotlib to see the plot: pip install matplotlib)")


if __name__ == "__main__":
    main()
