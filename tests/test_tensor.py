"""
tests/test_tensor.py — Autograd correctness tests via finite differences.

For every differentiable op we verify:
    numerical gradient ≈ analytical gradient

Numerical gradient uses the central difference formula:
    ∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)

Run with:
    python -m pytest tests/test_tensor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from neural_network_framework.tensor import Tensor


EPS = 1e-4       # finite difference step
ATOL = 1e-3      # absolute tolerance for grad comparison
RTOL = 1e-2      # relative tolerance


def numerical_gradient(func, x: np.ndarray, i: int) -> float:
    """Compute numerical gradient of scalar func w.r.t. x.flat[i]."""
    x_plus = x.copy().flatten()
    x_minus = x.copy().flatten()
    x_plus[i] += EPS
    x_minus[i] -= EPS
    x_plus = x_plus.reshape(x.shape)
    x_minus = x_minus.reshape(x.shape)
    return (func(x_plus) - func(x_minus)) / (2 * EPS)


def grad_check(func, x_np: np.ndarray, atol=ATOL):
    """
    Compare analytical vs numerical gradients for a scalar-valued function
    that takes a numpy array and returns a scalar.
    """
    x = Tensor(x_np.copy(), requires_grad=True)
    out = func(x)
    out.backward()
    analytical = x.grad.flatten()

    for i in range(x_np.size):
        num = numerical_gradient(lambda v: func(Tensor(v)).data.item(), x_np, i)
        assert abs(analytical[i] - num) < atol + RTOL * abs(num), (
            f"Gradient mismatch at index {i}: "
            f"analytical={analytical[i]:.6f}, numerical={num:.6f}"
        )


# -------------------------------------------------------------------
# Individual op tests
# -------------------------------------------------------------------

class TestBasicOps:
    def test_add(self):
        def f(x): return (x + Tensor(np.ones_like(x.data))).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_sub(self):
        def f(x): return (x - Tensor(np.ones_like(x.data))).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_mul(self):
        c = np.random.randn(3, 4).astype(np.float32)
        def f(x): return (x * Tensor(c)).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_div(self):
        c = np.random.randn(3, 4).astype(np.float32) + 2.0  # avoid near-zero
        def f(x): return (x / Tensor(c)).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_pow(self):
        def f(x): return (x ** 3).sum()
        # Shift away from zero so gradient magnitude is large enough for
        # relative tolerance comparison to work in float32.
        grad_check(f, np.abs(np.random.randn(3, 4).astype(np.float32)) + 1.0, atol=2e-3)

    def test_neg(self):
        def f(x): return (-x).sum()
        grad_check(f, np.random.randn(4).astype(np.float32))


class TestReductions:
    def test_sum(self):
        def f(x): return x.sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_sum_axis(self):
        def f(x): return x.sum(axis=0).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_mean(self):
        def f(x): return x.mean()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_mean_axis(self):
        def f(x): return x.mean(axis=1).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))


class TestMath:
    def test_exp(self):
        def f(x): return x.exp().sum()
        grad_check(f, np.random.randn(4).astype(np.float32) * 0.5)

    def test_log(self):
        def f(x): return x.log().sum()
        grad_check(f, np.abs(np.random.randn(4).astype(np.float32)) + 0.5)

    def test_sqrt(self):
        def f(x): return x.sqrt().sum()
        grad_check(f, np.abs(np.random.randn(4).astype(np.float32)) + 0.1)


class TestMatMul:
    def test_matmul_square(self):
        B = np.random.randn(4, 4).astype(np.float32)
        def f(x): return (x @ Tensor(B)).sum()
        # Float32 matmul accumulates rounding; loosen tolerance slightly.
        grad_check(f, np.random.randn(4, 4).astype(np.float32), atol=5e-3)

    def test_matmul_rect(self):
        B = np.random.randn(4, 3).astype(np.float32)
        def f(x): return (x @ Tensor(B)).sum()
        grad_check(f, np.random.randn(5, 4).astype(np.float32))


class TestShape:
    def test_reshape(self):
        def f(x): return x.reshape(2, 6).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_transpose(self):
        def f(x): return x.T.sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))


class TestChained:
    def test_chain(self):
        """Test a multi-op chain: z = mean((x @ W + b)^2)"""
        W = Tensor(np.random.randn(4, 3).astype(np.float32))
        b = Tensor(np.random.randn(3).astype(np.float32))
        def f(x): return ((x @ W + b) ** 2).mean()
        grad_check(f, np.random.randn(5, 4).astype(np.float32))

    def test_broadcast_add(self):
        b = np.random.randn(4).astype(np.float32)
        def f(x): return (x + Tensor(b)).sum()
        grad_check(f, np.random.randn(3, 4).astype(np.float32))

    def test_softmax_like(self):
        """Verify exp + sum + div gradient chain."""
        def f(x):
            e = x.exp()
            return (e / e.sum()).sum()
        grad_check(f, np.random.randn(5).astype(np.float32))

    def test_accumulate_grad(self):
        """Calling backward multiple times should accumulate gradients."""
        x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        out1 = x.sum()
        out2 = x.sum()
        out1.backward()
        out2.backward()
        # Each backward adds 1 to each grad → total grad should be [2, 2, 2]
        np.testing.assert_allclose(x.grad, np.array([2.0, 2.0, 2.0]))
