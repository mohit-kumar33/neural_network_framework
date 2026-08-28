"""
activations.py — Element-wise activation functions as Modules.

All activations are stateless modules (no learnable parameters).
The math for each derivative is annotated inline.
"""

from __future__ import annotations

import numpy as np

from neural_network_framework.tensor import Tensor
from neural_network_framework.nn.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit: f(x) = max(0, x)
    Gradient: 1 if x > 0 else 0
    """

    def forward(self, x: Tensor) -> Tensor:
        mask = (x.data > 0).astype(np.float32)
        out = Tensor(
            x.data * mask,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="ReLU",
        )

        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += out.grad * mask

        out._backward = _backward
        return out

    def extra_repr(self):
        return ""


class LeakyReLU(Module):
    """
    Leaky ReLU: f(x) = x if x > 0 else negative_slope * x
    Gradient: 1 if x > 0 else negative_slope
    """

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        slope = self.negative_slope
        mask_pos = (x.data > 0).astype(np.float32)
        mask_neg = 1.0 - mask_pos
        val = x.data * mask_pos + slope * x.data * mask_neg
        out = Tensor(val, requires_grad=x.requires_grad, _children=(x,), _op="LeakyReLU")

        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += out.grad * (mask_pos + slope * mask_neg)

        out._backward = _backward
        return out

    def extra_repr(self):
        return f"negative_slope={self.negative_slope}"


class Sigmoid(Module):
    """
    Sigmoid: f(x) = 1 / (1 + exp(-x))
    Gradient: f(x) * (1 - f(x))
    """

    def forward(self, x: Tensor) -> Tensor:
        sig = 1.0 / (1.0 + np.exp(-x.data.clip(-500, 500)))
        out = Tensor(sig, requires_grad=x.requires_grad, _children=(x,), _op="Sigmoid")

        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += out.grad * sig * (1.0 - sig)

        out._backward = _backward
        return out


class Tanh(Module):
    """
    Hyperbolic tangent: f(x) = tanh(x)
    Gradient: 1 - tanh(x)^2
    """

    def forward(self, x: Tensor) -> Tensor:
        val = np.tanh(x.data)
        out = Tensor(val, requires_grad=x.requires_grad, _children=(x,), _op="Tanh")

        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += out.grad * (1.0 - val ** 2)

        out._backward = _backward
        return out


class GELU(Module):
    """
    Gaussian Error Linear Unit (approximation used by GPT-2):
    f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Gradient computed via chain rule on the above formula.
    """

    _SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi).astype(np.float32)
    _COEFF = np.float32(0.044715)

    def forward(self, x: Tensor) -> Tensor:
        c = self._SQRT_2_OVER_PI
        k = self._COEFF
        inner = c * (x.data + k * x.data ** 3)
        tanh_val = np.tanh(inner)
        val = 0.5 * x.data * (1.0 + tanh_val)
        out = Tensor(val, requires_grad=x.requires_grad, _children=(x,), _op="GELU")

        def _backward():
            if x.requires_grad:
                x._init_grad()
                # d/dx [ 0.5*x*(1+tanh(inner)) ]
                dtanh = 1.0 - tanh_val ** 2  # sech^2
                dinner_dx = c * (1.0 + 3.0 * k * x.data ** 2)
                grad_x = 0.5 * (1.0 + tanh_val) + 0.5 * x.data * dtanh * dinner_dx
                x.grad += out.grad * grad_x

        out._backward = _backward
        return out


class Softmax(Module):
    """
    Softmax along a given axis (default: last axis).

    Note: Softmax is rarely used as an intermediate activation. It's
    typically baked into CrossEntropyLoss for numerical stability.
    """

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        # Subtract max for numerical stability
        shifted = x.data - x.data.max(axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted)
        sm = exp_x / exp_x.sum(axis=self.axis, keepdims=True)
        out = Tensor(sm, requires_grad=x.requires_grad, _children=(x,), _op="Softmax")

        def _backward():
            if x.requires_grad:
                x._init_grad()
                # dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
                dot = (out.grad * sm).sum(axis=self.axis, keepdims=True)
                x.grad += sm * (out.grad - dot)

        out._backward = _backward
        return out

    def extra_repr(self):
        return f"axis={self.axis}"


class LogSoftmax(Module):
    """
    Log-Softmax: log(softmax(x)) — numerically stable formulation.
    Commonly used with NLLLoss.
    """

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        shifted = x.data - x.data.max(axis=self.axis, keepdims=True)
        log_sm = shifted - np.log(np.exp(shifted).sum(axis=self.axis, keepdims=True))
        out = Tensor(log_sm, requires_grad=x.requires_grad, _children=(x,), _op="LogSoftmax")

        def _backward():
            if x.requires_grad:
                x._init_grad()
                sm = np.exp(log_sm)
                x.grad += out.grad - sm * out.grad.sum(axis=self.axis, keepdims=True)

        out._backward = _backward
        return out

    def extra_repr(self):
        return f"axis={self.axis}"
