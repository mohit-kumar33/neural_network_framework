"""
base.py — Abstract Optimizer base class.
"""

from __future__ import annotations

from neural_network_framework.tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers.

    Subclasses must implement `step()`.

    Parameters
    ----------
    params : list[Tensor]
        List of Tensors to optimise (typically `model.parameters()`).
    """

    def __init__(self, params: list[Tensor]):
        if not params:
            raise ValueError("Optimizer requires at least one parameter.")
        self.params = list(params)

    def zero_grad(self):
        """Reset gradients of all managed parameters to zero."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        """Perform a single optimisation step. Must be overridden."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={len(self.params)})"
