"""
examples/02_mnist.py — MNIST handwritten digit classifier.

Uses scikit-learn to download the dataset (70 000 samples, 784 features).
Trains a 3-layer MLP to ~97% test accuracy.

Requirements:
    pip install scikit-learn

Run:
    python examples/02_mnist.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import neural_network_framework.nn as nn
from neural_network_framework.tensor import Tensor
from neural_network_framework.optim import Adam
from neural_network_framework.loss import CrossEntropyLoss
from neural_network_framework.utils.data import TensorDataset, DataLoader


def load_mnist():
    """Download and preprocess MNIST via scikit-learn."""
    print("Downloading MNIST (this may take a moment on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0  # normalise to [0, 1]
    y = mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10_000, random_state=42, stratify=y
    )

    # Zero-mean, unit-variance normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test


def load_synthetic():
    """
    Fallback: a synthetic multi-class problem when sklearn is not available.
    Creates 10 classes, 100 features, 5000 samples.
    """
    print("(scikit-learn not found — using synthetic 10-class data)")
    np.random.seed(42)
    N, C, D = 5000, 10, 100
    centres = np.random.randn(C, D).astype(np.float32) * 3
    X = np.vstack([centres[c] + 0.5 * np.random.randn(N // C, D)
                   for c in range(C)]).astype(np.float32)
    y = np.repeat(np.arange(C), N // C)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:]


def accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = logits.data.argmax(axis=1)
    return (preds == targets.data).mean()


def main():
    np.random.seed(42)

    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = load_mnist()
        in_features = 784
        num_classes = 10
    else:
        X_train, X_test, y_train, y_test = load_synthetic()
        in_features = X_train.shape[1]
        num_classes = 10

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # 3-layer MLP with BatchNorm and Dropout
    model = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("  MNIST Classifier — 3-layer MLP")
    print(f"  Parameters: {sum(p.data.size for p in model.parameters()):,}")
    print("=" * 60)
    print(f"{'Epoch':>5}  {'Train Loss':>12}  {'Train Acc':>10}  {'Test Acc':>10}")
    print("-" * 45)

    for epoch in range(1, 11):
        # --- Training ---
        model.train()
        train_losses, train_accs = [], []

        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(accuracy(logits, y_batch))

        # --- Evaluation ---
        model.eval()
        test_accs = []
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            test_accs.append(accuracy(logits, y_batch))

        print(
            f"{epoch:>5}  "
            f"{np.mean(train_losses):>12.4f}  "
            f"{np.mean(train_accs) * 100:>9.2f}%  "
            f"{np.mean(test_accs) * 100:>9.2f}%"
        )

    print("-" * 45)
    print("\nDone! Final test accuracy: "
          f"{np.mean(test_accs) * 100:.2f}%")


if __name__ == "__main__":
    main()
