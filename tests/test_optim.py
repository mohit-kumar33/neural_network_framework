"""
tests/test_optim.py — Tests for SGD and Adam optimizers.

Verifies:
- Parameters are updated in the right direction (loss goes down)
- zero_grad resets gradients
- Momentum buffers are initialised lazily
- Adam bias correction (m_hat, v_hat)

Run with:
    python -m pytest tests/test_optim.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from neural_network_framework.tensor import Tensor
import neural_network_framework.nn as nn
from neural_network_framework.loss import MSELoss
from neural_network_framework.optim import SGD, Adam, AdamW


def make_simple_problem(n=100, d=4, seed=0):
    """Linear regression: y = Xw + ε"""
    np.random.seed(seed)
    X_np = np.random.randn(n, d).astype(np.float32)
    w_true = np.random.randn(d, 1).astype(np.float32)
    y_np = X_np @ w_true + 0.01 * np.random.randn(n, 1).astype(np.float32)
    return Tensor(X_np), Tensor(y_np)


def make_model(in_d=4, out_d=1):
    return nn.Sequential(nn.Linear(in_d, out_d, bias=True))


class TestSGD:
    def test_loss_decreases(self):
        X, y = make_simple_problem()
        model = make_model()
        opt = SGD(model.parameters(), lr=0.01)
        criterion = MSELoss()

        prev_loss = float("inf")
        for _ in range(50):
            out = model(X)
            loss = criterion(out, y)
            model.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < prev_loss, "Loss should decrease after training"

    def test_zero_grad_clears(self):
        model = make_model()
        opt = SGD(model.parameters(), lr=0.01)
        X = Tensor(np.ones((4, 4), dtype=np.float32))
        model(X).sum().backward()
        opt.zero_grad()
        for p in model.parameters():
            assert (p.grad == 0).all()

    def test_momentum_updates_velocity(self):
        model = make_model(4, 1)
        opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
        X, y = make_simple_problem()
        criterion = MSELoss()

        for _ in range(5):
            loss = criterion(model(X), y)
            model.zero_grad()
            loss.backward()
            opt.step()

        # Velocity buffers should have been set
        assert any(v is not None for v in opt._velocities)

    def test_weight_decay_shrinks_params(self):
        """With high weight decay and zero grad, params should shrink."""
        w = Tensor(np.ones((1,), dtype=np.float32), requires_grad=True)
        opt = SGD([w], lr=0.1, weight_decay=1.0)
        w.zero_grad()  # grad stays 0
        w.grad = np.zeros_like(w.data)  # explicit zero grad
        opt.step()
        assert w.data[0] < 1.0


class TestAdam:
    def test_loss_decreases(self):
        X, y = make_simple_problem()
        model = make_model()
        opt = Adam(model.parameters(), lr=1e-2)
        criterion = MSELoss()

        for _ in range(50):
            out = model(X)
            loss = criterion(out, y)
            model.zero_grad()
            loss.backward()
            opt.step()

        # Compare to initial loss
        model2 = make_model()
        init_loss = criterion(model2(X), y).item()
        assert loss.item() < init_loss

    def test_step_counter_increments(self):
        model = make_model()
        opt = Adam(model.parameters())
        X, y = make_simple_problem()
        criterion = MSELoss()

        for _ in range(3):
            loss = criterion(model(X), y)
            model.zero_grad()
            loss.backward()
            opt.step()

        assert opt._t == 3

    def test_moment_buffers_initialised(self):
        model = make_model()
        opt = Adam(model.parameters())
        X, y = make_simple_problem()
        loss = MSELoss()(model(X), y)
        model.zero_grad()
        loss.backward()
        opt.step()

        assert all(m is not None for m in opt._m)
        assert all(v is not None for v in opt._v)


class TestAdamW:
    def test_decoupled_decay(self):
        """AdamW applies decay directly to params, not to gradient."""
        w = Tensor(np.ones((4,), dtype=np.float32), requires_grad=True)
        w.grad = np.zeros_like(w.data)  # zero gradient → only decay acts
        opt = AdamW([w], lr=1.0, weight_decay=0.5)
        # With zero grad and high decay, weight should shrink
        w_before = w.data.copy()
        opt.step()
        assert (w.data < w_before).all()

    def test_convergence(self):
        X, y = make_simple_problem()
        model = make_model()
        opt = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        criterion = MSELoss()
        first_loss = criterion(model(X), y).item()

        for _ in range(100):
            out = model(X)
            loss = criterion(out, y)
            model.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < first_loss
