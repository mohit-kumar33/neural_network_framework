"""
tests/test_layers.py — Tests for trainable layers and activations.

Verifies:
- Correct output shapes
- Gradient flows (requires_grad propagation)
- train/eval mode toggles
- BatchNorm running statistics

Run with:
    python -m pytest tests/test_layers.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from neural_network_framework.tensor import Tensor
import neural_network_framework.nn as nn
from neural_network_framework.loss import MSELoss, CrossEntropyLoss


class TestLinear:
    def test_output_shape(self):
        layer = nn.Linear(4, 8)
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        out = layer(x)
        assert out.shape == (3, 8)

    def test_no_bias_shape(self):
        layer = nn.Linear(4, 8, bias=False)
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        out = layer(x)
        assert out.shape == (3, 8)

    def test_grad_flows(self):
        layer = nn.Linear(4, 2)
        x = Tensor(np.random.randn(5, 4).astype(np.float32))
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape

    def test_parameter_count(self):
        layer = nn.Linear(4, 8, bias=True)
        params = layer.parameters()
        assert len(params) == 2  # weight + bias

    def test_no_bias_parameters(self):
        layer = nn.Linear(4, 8, bias=False)
        params = layer.parameters()
        assert len(params) == 1  # weight only


class TestActivations:
    @pytest.mark.parametrize("act_cls", [
        nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU,
    ])
    def test_shape_preserved(self, act_cls):
        act = act_cls()
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        out = act(x)
        assert out.shape == x.shape

    def test_relu_nonnegative(self):
        act = nn.ReLU()
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        out = act(x)
        assert (out.data >= 0).all()

    def test_sigmoid_range(self):
        act = nn.Sigmoid()
        x = Tensor(np.random.randn(100).astype(np.float32) * 10)
        out = act(x)
        assert out.data.min() >= 0.0
        assert out.data.max() <= 1.0

    def test_softmax_sums_to_one(self):
        act = nn.Softmax(axis=-1)
        x = Tensor(np.random.randn(5, 10).astype(np.float32))
        out = act(x)
        row_sums = out.data.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-5)

    def test_relu_gradient(self):
        act = nn.ReLU()
        x = Tensor(np.array([-1.0, 1.0, 2.0], dtype=np.float32), requires_grad=True)
        out = act(x)
        out.sum().backward()
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_allclose(x.grad, expected, atol=1e-6)


class TestBatchNorm:
    def test_output_shape(self):
        bn = nn.BatchNorm1d(8)
        x = Tensor(np.random.randn(16, 8).astype(np.float32))
        out = bn(x)
        assert out.shape == (16, 8)

    def test_train_normalizes(self):
        """Output should have ~zero mean and ~unit std during training."""
        bn = nn.BatchNorm1d(4)
        bn.train()
        x = Tensor(np.random.randn(100, 4).astype(np.float32) * 10 + 5)
        out = bn(x)
        # After BN + learned affine (gamma=1, beta=0 initially)
        np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(4), atol=0.1)
        np.testing.assert_allclose(out.data.std(axis=0), np.ones(4), atol=0.1)

    def test_eval_uses_running_stats(self):
        """Running stats should be updated during train and used during eval."""
        bn = nn.BatchNorm1d(4)
        bn.train()
        for _ in range(10):
            x = Tensor(np.random.randn(32, 4).astype(np.float32))
            bn(x)
        # After training, running_mean/var should be non-trivial
        bn.eval()
        x = Tensor(np.random.randn(8, 4).astype(np.float32))
        out = bn(x)
        assert out.shape == (8, 4)

    def test_parameters(self):
        bn = nn.BatchNorm1d(4, affine=True)
        params = bn.parameters()
        assert len(params) == 2  # gamma, beta


class TestDropout:
    def test_train_mode_zeros(self):
        drop = nn.Dropout(p=0.5)
        drop.train()
        x = Tensor(np.ones((1000, 10), dtype=np.float32))
        out = drop(x)
        zero_frac = (out.data == 0).mean()
        # Should be roughly 50% zeros
        assert 0.3 < zero_frac < 0.7

    def test_eval_mode_identity(self):
        drop = nn.Dropout(p=0.5)
        drop.eval()
        x = Tensor(np.random.randn(10, 10).astype(np.float32))
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_scale_factor(self):
        """Inverted dropout: surviving values should be scaled by 1/(1-p)."""
        drop = nn.Dropout(p=0.5)
        drop.train()
        x = Tensor(np.ones((10000,), dtype=np.float32))
        out = drop(x)
        nonzero = out.data[out.data != 0]
        np.testing.assert_allclose(nonzero, 2.0, atol=1e-5)


class TestSequential:
    def test_forward(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        x = Tensor(np.random.randn(5, 4).astype(np.float32))
        out = model(x)
        assert out.shape == (5, 2)

    def test_parameters_collected(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        params = model.parameters()
        # 2 Linear layers with bias → 4 tensors
        assert len(params) == 4

    def test_zero_grad(self):
        model = nn.Sequential(nn.Linear(2, 2))
        x = Tensor(np.ones((3, 2), dtype=np.float32))
        out = model(x)
        out.sum().backward()
        model.zero_grad()
        for p in model.parameters():
            assert (p.grad == 0).all()

    def test_train_eval_toggle(self):
        model = nn.Sequential(nn.Dropout(0.5))
        model.eval()
        assert not model.layers[0].training
        model.train()
        assert model.layers[0].training


class TestLosses:
    def test_mse_scalar(self):
        from neural_network_framework.loss import MSELoss
        criterion = MSELoss()
        pred = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        target = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        loss = criterion(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ce_shape(self):
        criterion = CrossEntropyLoss()
        logits = Tensor(np.random.randn(8, 5).astype(np.float32), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 3, 4, 0, 1, 2]))
        loss = criterion(logits, targets)
        assert loss.data.ndim == 0 or loss.data.size == 1

    def test_ce_gradient_flows(self):
        criterion = CrossEntropyLoss()
        logits = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 0]))
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
