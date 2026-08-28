"""
layers.py — Trainable neural network layers.

Each layer stores its parameters as Tensors with requires_grad=True.
The Module base class automatically discovers them via `parameters()`.
"""

from __future__ import annotations

import numpy as np

from neural_network_framework.tensor import Tensor
from neural_network_framework.nn.module import Module
from neural_network_framework.utils.init import xavier_uniform, he_uniform


class Linear(Module):
    """
    Fully-connected (dense) layer: y = x @ W^T + b

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If False, no additive bias term is used.
    init : str
        Weight initialisation scheme — 'he' (default for ReLU networks)
        or 'xavier' (recommended for tanh/sigmoid networks).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "he",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Weight matrix: shape (out_features, in_features)
        if init == "xavier":
            w_data = xavier_uniform(in_features, out_features)
        else:  # he / kaiming
            w_data = he_uniform(in_features, out_features)

        self.weight = Tensor(w_data, requires_grad=True)

        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features)  →  out: (..., out_features)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.use_bias}"
        )


class BatchNorm1d(Module):
    """
    Batch Normalisation over a 2D input (batch of 1-D feature vectors).

    During training: normalises using batch statistics (μ, σ²).
    During eval:     normalises using running statistics.

    Parameters
    ----------
    num_features : int
        Number of features (C) in the input (shape: N×C).
    eps : float
        Value added to the denominator for numerical stability.
    momentum : float
        Factor for the running mean/var update (like PyTorch's BatchNorm).
    affine : bool
        If True, learnable scale (γ) and shift (β) are added.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        else:
            self.gamma = None
            self.beta = None

        # Running statistics (not parameters — not returned by parameters())
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Batch statistics
            mu = x.data.mean(axis=0)
            var = x.data.var(axis=0)

            # Update running stats
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mu
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            )
        else:
            mu = self.running_mean
            var = self.running_var

        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat_data = (x.data - mu) * inv_std

        out = Tensor(
            x_hat_data,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="BatchNorm1d",
        )

        # Cache for backward
        _mu, _inv_std, _x_hat = mu, inv_std, x_hat_data
        _N = x.data.shape[0]
        _training = self.training

        def _backward():
            if x.requires_grad:
                x._init_grad()
                dout = out.grad  # (N, C)
                if _training:
                    # Full BN backward (Ioffe & Szegedy 2015)
                    dx_hat = dout
                    dvar = (-0.5 * (dx_hat * _x_hat).sum(axis=0) * _inv_std ** 2)
                    dmu = (-dx_hat * _inv_std).sum(axis=0) + dvar * (-2.0 / _N) * (x.data - _mu)
                    dx = (dx_hat * _inv_std
                          + dvar * 2.0 * (x.data - _mu) / _N
                          + dmu / _N)
                else:
                    dx = dout * _inv_std
                x.grad += dx

        out._backward = _backward

        # Apply affine transform
        if self.affine and self.gamma is not None and self.beta is not None:
            out = self.gamma * out + self.beta

        return out

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}"
        )


class Dropout(Module):
    """
    Randomly zeroes elements of the input with probability `p` during training.
    Scales the remaining elements by 1/(1-p) (inverted dropout).

    Parameters
    ----------
    p : float
        Probability of zeroing an element. Default: 0.5.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        scale = 1.0 / (1.0 - self.p)
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) * scale
        out = Tensor(
            x.data * mask,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="Dropout",
        )

        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += out.grad * mask

        out._backward = _backward
        return out

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Embedding(Module):
    """
    A lookup table for dense vector representations.

    Parameters
    ----------
    num_embeddings : int
        Size of the dictionary (vocabulary size).
    embedding_dim : int
        Dimensionality of each embedding vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialise with small normal values
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True,
        )

    def forward(self, indices) -> Tensor:
        """
        Parameters
        ----------
        indices : array-like of int, shape (*)
            Indices into the embedding table.
        """
        if isinstance(indices, Tensor):
            idx = indices.data.astype(int)
        else:
            idx = np.array(indices, dtype=int)

        selected = self.weight.data[idx]  # (*embedding_dim)
        out = Tensor(selected, requires_grad=self.weight.requires_grad, _children=(self.weight,), _op="Embedding")

        def _backward():
            if self.weight.requires_grad:
                self.weight._init_grad()
                np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
