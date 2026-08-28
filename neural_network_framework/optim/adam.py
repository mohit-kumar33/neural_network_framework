"""
adam.py — Adam and AdamW optimizers.

Adam (Kingma & Ba, 2014):
    m_t = β1 * m_{t-1} + (1 - β1) * g_t          (1st moment / mean)
    v_t = β2 * v_{t-1} + (1 - β2) * g_t^2         (2nd moment / variance)
    m̂_t = m_t / (1 - β1^t)                        (bias-corrected)
    v̂_t = v_t / (1 - β2^t)                        (bias-corrected)
    θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

AdamW decouples weight decay from the gradient update, which is generally
superior to adding λθ to the gradient (as Adam does by default).
"""

from __future__ import annotations

import numpy as np

from neural_network_framework.tensor import Tensor
from neural_network_framework.optim.base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Parameters
    ----------
    params : list[Tensor]
    lr : float
        Learning rate (α). Default: 1e-3.
    betas : tuple[float, float]
        Coefficients (β1, β2) for computing running averages of
        gradient and its square. Default: (0.9, 0.999).
    eps : float
        Term added to denominator for numerical stability. Default: 1e-8.
    weight_decay : float
        L2 regularisation (added to gradient). Default: 0.0.
        For decoupled weight decay, use AdamW instead.
    amsgrad : bool
        Whether to use the AMSGrad variant. Default: False.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        n = len(self.params)
        self._m: list[np.ndarray | None] = [None] * n   # 1st moment
        self._v: list[np.ndarray | None] = [None] * n   # 2nd moment
        self._v_max: list[np.ndarray | None] = [None] * n  # AMSGrad
        self._t: int = 0                                  # step counter

    def step(self):
        self._t += 1
        t = self._t
        b1, b2, eps = self.beta1, self.beta2, self.eps

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            # Initialise moment buffers
            if self._m[i] is None:
                self._m[i] = np.zeros_like(p.data)
                self._v[i] = np.zeros_like(p.data)
                if self.amsgrad:
                    self._v_max[i] = np.zeros_like(p.data)

            # Update biased moment estimates
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g ** 2)

            # Bias-corrected estimates
            m_hat = self._m[i] / (1 - b1 ** t)
            v_hat = self._v[i] / (1 - b2 ** t)

            if self.amsgrad:
                self._v_max[i] = np.maximum(self._v_max[i], v_hat)
                v_hat = self._v_max[i]

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1},{self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )


class AdamW(Optimizer):
    """
    AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Unlike Adam, weight decay is applied directly to the parameters
    (not to the gradient), which prevents the adaptive learning rate
    from reducing the effective regularisation.

    Parameters
    ----------
    params, lr, betas, eps : same as Adam.
    weight_decay : float
        Decoupled weight decay coefficient. Default: 0.01.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        n = len(self.params)
        self._m: list[np.ndarray | None] = [None] * n
        self._v: list[np.ndarray | None] = [None] * n
        self._t: int = 0

    def step(self):
        self._t += 1
        t = self._t
        b1, b2, eps = self.beta1, self.beta2, self.eps

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.copy()

            if self._m[i] is None:
                self._m[i] = np.zeros_like(p.data)
                self._v[i] = np.zeros_like(p.data)

            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g ** 2)

            m_hat = self._m[i] / (1 - b1 ** t)
            v_hat = self._v[i] / (1 - b2 ** t)

            # Decoupled weight decay (applied directly to params, not to grad)
            p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1},{self.beta2}), "
            f"weight_decay={self.weight_decay})"
        )
