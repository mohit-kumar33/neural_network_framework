"""
init.py — Weight initialisation strategies.

Proper initialisation is critical to training stability. This module
provides Xavier (for tanh/sigmoid) and He/Kaiming (for ReLU) initialisers.
"""

from __future__ import annotations

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Xavier / Glorot uniform initialisation.

    Draws from Uniform(-limit, limit) where
    limit = sqrt(6 / (fan_in + fan_out))

    Recommended for tanh and sigmoid activations.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32)


def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Xavier / Glorot normal initialisation.

    Draws from Normal(0, std) where
    std = sqrt(2 / (fan_in + fan_out))
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_out, fan_in).astype(np.float32) * std


def he_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """
    He / Kaiming uniform initialisation.

    Draws from Uniform(-limit, limit) where
    limit = sqrt(6 / fan_in)

    Recommended for ReLU activations.
    """
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32)


def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """
    He / Kaiming normal initialisation.

    Draws from Normal(0, std) where
    std = sqrt(2 / fan_in)
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_out, fan_in).astype(np.float32) * std


def orthogonal(rows: int, cols: int, gain: float = 1.0) -> np.ndarray:
    """
    Orthogonal initialisation via SVD.
    Useful for RNNs and very deep networks.
    """
    flat = np.random.randn(max(rows, cols), min(rows, cols)).astype(np.float32)
    U, _, Vt = np.linalg.svd(flat, full_matrices=False)
    out = U if rows >= cols else Vt
    return (gain * out[:rows, :cols]).astype(np.float32)
