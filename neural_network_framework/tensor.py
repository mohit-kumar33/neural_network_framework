"""
tensor.py — The core Tensor class with automatic differentiation.

Every Tensor wraps a NumPy ndarray. Arithmetic operations record a backward
closure on the output so that calling `tensor.backward()` propagates gradients
through the entire computation graph via reverse-mode autodiff (backprop).

Design principles
-----------------
* No magic — every line of gradient math is written explicitly.
* Lazy gradient accumulation: grad is None until backward() is called.
* Works with arbitrary batch sizes (leading dimensions are treated as batch).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Set


class Tensor:
    """A multi-dimensional array with automatic differentiation support."""

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        dtype=np.float32,
    ):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)

        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None  # accumulated gradient

        # Internal autograd bookkeeping
        self._backward = lambda: None   # local backward closure
        self._prev: Set[Tensor] = set(_children)
        self._op: str = _op             # for debugging / printing

    # ------------------------------------------------------------------
    # Properties / dunder helpers
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __repr__(self) -> str:
        grad_info = f", grad_fn=<{self._op}>" if self._op else ""
        return f"Tensor({self.data}{grad_info})"

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Reset gradient to zero (keeps the same array in memory)."""
        if self.grad is not None:
            self.grad.fill(0.0)
        else:
            self.grad = np.zeros_like(self.data)

    def _init_grad(self):
        """Lazily initialise gradient storage."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Compute gradients of this tensor w.r.t. all leaf tensors that
        have requires_grad=True, using reverse-mode autodiff.

        Parameters
        ----------
        grad : np.ndarray or None
            Upstream gradient. If None the tensor is assumed to be a scalar
            and grad is set to 1.0.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() can only be called without a gradient argument "
                    "on scalar tensors. Pass an explicit gradient for non-scalar outputs."
                )
            grad = np.ones_like(self.data)

        self._init_grad()
        self.grad += grad

        # Build topological order
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(node: Tensor):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Walk in reverse topological order
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor | float | int") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="Add",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: "Tensor | float | int") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="Sub",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other) -> "Tensor":
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other: "Tensor | float | int") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="Mul",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: "Tensor | float | int") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="Div",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(
                    -out.grad * self.data / (other.data ** 2), other.shape
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        return _ensure_tensor(other).__truediv__(self)

    def __pow__(self, exponent: float) -> "Tensor":
        assert isinstance(exponent, (int, float)), "Only scalar exponents are supported."
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Pow",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="MatMul",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                # dL/dA = dL/dC @ B^T
                g = out.grad @ other.data.swapaxes(-1, -2)
                self.grad += _unbroadcast(g, self.shape)
            if other.requires_grad:
                other._init_grad()
                # dL/dB = A^T @ dL/dC
                g = self.data.swapaxes(-1, -2) @ out.grad
                other.grad += _unbroadcast(g, other.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reduction operations
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Sum",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad
                # Restore reduced axes so broadcasting works
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Mean",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                n = self.data.size if axis is None else self.data.shape[axis]
                self.grad += np.broadcast_to(grad / n, self.shape)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        result = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(
            result,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Max",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                val = result
                if axis is not None and not keepdims:
                    val = np.expand_dims(val, axis=axis)
                mask = (self.data == np.broadcast_to(val, self.shape)).astype(np.float32)
                # Normalise ties
                mask /= mask.sum(axis=axis, keepdims=True) + 1e-10
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape) * mask

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Element-wise math
    # ------------------------------------------------------------------

    def exp(self) -> "Tensor":
        val = np.exp(self.data)
        out = Tensor(
            val,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Exp",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * val

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(np.clip(self.data, 1e-7, None)),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Log",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad / np.clip(self.data, 1e-7, None)

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def abs(self) -> "Tensor":
        out = Tensor(
            np.abs(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Abs",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * np.sign(self.data)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def reshape(self, *shape) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Reshape",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None) -> "Tensor":
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="Transpose",
        )

        def _backward():
            if self.requires_grad:
                self._init_grad()
                if axes is None:
                    self.grad += out.grad.transpose()
                else:
                    inv = np.argsort(axes)
                    self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    def flatten(self) -> "Tensor":
        return self.reshape(self.shape[0], -1)

    # ------------------------------------------------------------------
    # Comparison / indexing helpers (no grad)
    # ------------------------------------------------------------------

    def argmax(self, axis=None) -> np.ndarray:
        return self.data.argmax(axis=axis)

    def item(self) -> float:
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    # ------------------------------------------------------------------
    # Factory / class methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(cls, *shape, requires_grad=False, dtype=np.float32) -> "Tensor":
        return cls(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, requires_grad=False, dtype=np.float32) -> "Tensor":
        return cls(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, requires_grad=False, dtype=np.float32) -> "Tensor":
        return cls(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, requires_grad=False) -> "Tensor":
        return cls(arr.copy(), requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_tensor(x) -> Tensor:
    """Wrap scalars/arrays in a Tensor if needed."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sum gradient over axes that were broadcast during a forward op so that
    the resulting gradient has the same shape as `target_shape`.
    """
    if grad.shape == target_shape:
        return grad

    # If target is a scalar
    if len(target_shape) == 0:
        return grad.sum(keepdims=False)

    # Pad target shape on the left with 1s to match ndim
    ndim_diff = grad.ndim - len(target_shape)
    padded = (1,) * ndim_diff + target_shape

    # Sum over axes where target has size 1
    sum_axes = tuple(i for i, s in enumerate(padded) if s == 1)
    if sum_axes:
        grad = grad.sum(axis=sum_axes, keepdims=True)

    # Remove leading dimensions added by padding
    if ndim_diff > 0:
        grad = grad.reshape(target_shape)

    return grad
