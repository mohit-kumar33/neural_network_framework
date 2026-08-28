"""
loss.py — Loss functions for training neural networks.

All losses are callable Modules and return a scalar Tensor.
Gradients flow back through them into the model parameters.
"""

from __future__ import annotations

import numpy as np

from neural_network_framework.tensor import Tensor
from neural_network_framework.nn.module import Module


class MSELoss(Module):
    """
    Mean Squared Error: L = mean((pred - target)^2)

    Used for regression tasks.

    Parameters
    ----------
    reduction : str
        'mean' (default) or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq = diff ** 2
        return sq.mean() if self.reduction == "mean" else sq.sum()

    def extra_repr(self):
        return f"reduction='{self.reduction}'"


class BCELoss(Module):
    """
    Binary Cross-Entropy Loss (with sigmoid applied externally):
    L = -mean(y * log(p) + (1-y) * log(1-p))

    Parameters
    ----------
    reduction : str
        'mean' (default) or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
        self._eps = 1e-7

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = self._eps
        p = pred.data.clip(eps, 1 - eps)
        y = target.data

        loss_data = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        if self.reduction == "mean":
            loss_val = loss_data.mean()
        else:
            loss_val = loss_data.sum()

        out = Tensor(loss_val, requires_grad=pred.requires_grad, _children=(pred,), _op="BCELoss")

        def _backward():
            if pred.requires_grad:
                pred._init_grad()
                grad = -(y / p - (1 - y) / (1 - p))
                if self.reduction == "mean":
                    grad /= pred.data.size
                pred.grad += out.grad * grad

        out._backward = _backward
        return out

    def extra_repr(self):
        return f"reduction='{self.reduction}'"


class BCEWithLogitsLoss(Module):
    """
    Numerically stable BCE: combines sigmoid + BCE in a single op.
    L = mean(max(x, 0) - x*y + log(1 + exp(-|x|)))

    Preferred over BCELoss + Sigmoid for stability.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        x = logits.data
        y = target.data
        loss_data = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))

        if self.reduction == "mean":
            loss_val = loss_data.mean()
        else:
            loss_val = loss_data.sum()

        out = Tensor(loss_val, requires_grad=logits.requires_grad, _children=(logits,), _op="BCEWithLogitsLoss")

        def _backward():
            if logits.requires_grad:
                logits._init_grad()
                sig = 1.0 / (1.0 + np.exp(-x))
                grad = sig - y
                if self.reduction == "mean":
                    grad /= x.size
                logits.grad += out.grad * grad

        out._backward = _backward
        return out


class CrossEntropyLoss(Module):
    """
    Cross-Entropy Loss for multi-class classification.

    Combines LogSoftmax + NLLLoss in a single numerically-stable pass.

    Parameters
    ----------
    reduction : str
        'mean' (default) or 'sum'.

    Input shapes
    ------------
    logits : (N, C)  — raw unnormalised scores (do NOT pre-softmax)
    target : (N,)    — class indices in [0, C)
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        N, C = logits.data.shape
        y = target.data.astype(int).flatten()

        # Numerically stable log-softmax
        shifted = logits.data - logits.data.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp  # (N, C)

        # NLL for the correct class
        nll = -log_probs[np.arange(N), y]  # (N,)
        loss_val = nll.mean() if self.reduction == "mean" else nll.sum()

        out = Tensor(
            loss_val,
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op="CrossEntropyLoss",
        )

        def _backward():
            if logits.requires_grad:
                logits._init_grad()
                # dL/d_logits = softmax(logits) - one_hot(target)
                sm = np.exp(log_probs)  # softmax values
                grad = sm.copy()
                grad[np.arange(N), y] -= 1.0
                if self.reduction == "mean":
                    grad /= N
                logits.grad += out.grad * grad

        out._backward = _backward
        return out

    def extra_repr(self):
        return f"reduction='{self.reduction}'"


class NLLLoss(Module):
    """
    Negative Log-Likelihood Loss.
    Expects log-probabilities as input (use with LogSoftmax).

    Input shapes
    ------------
    log_probs : (N, C)
    target    : (N,) — class indices
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        N = log_probs.data.shape[0]
        y = target.data.astype(int).flatten()
        nll = -log_probs.data[np.arange(N), y]
        loss_val = nll.mean() if self.reduction == "mean" else nll.sum()

        out = Tensor(loss_val, requires_grad=log_probs.requires_grad, _children=(log_probs,), _op="NLLLoss")

        def _backward():
            if log_probs.requires_grad:
                log_probs._init_grad()
                grad = np.zeros_like(log_probs.data)
                grad[np.arange(N), y] = -1.0
                if self.reduction == "mean":
                    grad /= N
                log_probs.grad += out.grad * grad

        out._backward = _backward
        return out
