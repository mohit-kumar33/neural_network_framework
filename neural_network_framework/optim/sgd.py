"""
sgd.py — Stochastic Gradient Descent with momentum and weight decay.

Update rule (with momentum μ and weight decay λ):

    v_t = μ * v_{t-1} + (1 - μ) * g_t      [Nesterov: use v_t to compute g_t]
    θ_t = θ_{t-1} - lr * (v_t + λ * θ_{t-1})
"""

from __future__ import annotations

import numpy as np

from neural_network_framework.tensor import Tensor
from neural_network_framework.optim.base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (with optional momentum and weight decay).

    Parameters
    ----------
    params : list[Tensor]
        Parameters to optimise.
    lr : float
        Learning rate.
    momentum : float
        Momentum factor (0 = plain SGD). Default: 0.0.
    weight_decay : float
        L2 regularisation coefficient. Default: 0.0.
    nesterov : bool
        If True, Nesterov momentum is used. Requires momentum > 0.
        Default: False.
    dampening : float
        Dampening for momentum. Default: 0.0.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0,
    ):
        super().__init__(params)
        assert lr > 0, "Learning rate must be positive."
        assert 0.0 <= momentum < 1.0, "Momentum must be in [0, 1)."

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening

        # Velocity buffers — initialised lazily on first step
        self._velocities: list[np.ndarray | None] = [None] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.copy()

            # L2 weight decay (add λθ to gradient)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            if self.momentum != 0.0:
                v = self._velocities[i]
                if v is None:
                    # First step: initialise velocity
                    self._velocities[i] = g.copy()
                    v = self._velocities[i]
                else:
                    v = self.momentum * v + (1.0 - self.dampening) * g
                    self._velocities[i] = v

                if self.nesterov:
                    g = g + self.momentum * v
                else:
                    g = v

            p.data -= self.lr * g

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, nesterov={self.nesterov})"
        )
