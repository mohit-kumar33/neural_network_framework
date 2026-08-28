"""
module.py — Base class for all neural network modules.

Every layer, activation, and loss function inherits from Module.
The design mirrors PyTorch's nn.Module for familiarity, but the
implementation is completely transparent.
"""

from __future__ import annotations
from typing import Iterator, Dict, Any

from neural_network_framework.tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses must implement `forward(*args, **kwargs)`.

    Parameters and sub-modules registered as attributes are
    automatically discovered by `parameters()` and `zero_grad()`.
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    # ------------------------------------------------------------------
    # Parameter discovery
    # ------------------------------------------------------------------

    def parameters(self) -> list[Tensor]:
        """
        Recursively collect all Tensors that require gradients from
        this module and all registered sub-modules.
        """
        params: list[Tensor] = []
        seen: set[int] = set()

        def _collect(obj):
            if isinstance(obj, Tensor) and obj.requires_grad:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    params.append(obj)
            elif isinstance(obj, Module):
                for v in vars(obj).values():
                    _collect(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    _collect(item)

        _collect(self)
        return params

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """Yield (name, tensor) pairs for all parameters."""

        def _walk(module: Module, prefix: str):
            for name, val in vars(module).items():
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(val, Tensor) and val.requires_grad:
                    yield full, val
                elif isinstance(val, Module):
                    yield from _walk(val, full)
                elif isinstance(val, (list, tuple)):
                    for i, item in enumerate(val):
                        if isinstance(item, Module):
                            yield from _walk(item, f"{full}[{i}]")

        yield from _walk(self, "")

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for p in self.parameters():
            p.zero_grad()

    # ------------------------------------------------------------------
    # Training / eval mode (used by Dropout, BatchNorm)
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Set this module and all sub-modules to training mode."""
        self.training = mode
        for v in vars(self).values():
            if isinstance(v, Module):
                v.train(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        """Set this module and all sub-modules to evaluation mode."""
        return self.train(False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Inject a default `training` attribute so subclasses don't have to.
        original_init = cls.__init__ if "__init__" in cls.__dict__ else None

        def patched_init(self, *args, **kwargs):
            self.training = True
            if original_init:
                original_init(self, *args, **kwargs)

        cls.__init__ = patched_init

    def extra_repr(self) -> str:
        """Override to add extra info in __repr__."""
        return ""

    def __repr__(self) -> str:
        extra = self.extra_repr()
        lines = [f"{self.__class__.__name__}({extra}"]
        for name, val in vars(self).items():
            if isinstance(val, Module):
                lines.append(f"  ({name}): {repr(val)}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """
    A sequential container. Modules are executed in the order they
    are passed to the constructor.

    Example
    -------
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )
    """

    def __init__(self, *modules: Module):
        self.layers = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        seen: set[int] = set()
        for layer in self.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def train(self, mode: bool = True):
        self.training = mode
        for layer in self.layers:
            layer.train(mode)
        return self

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {repr(layer)}")
        lines.append(")")
        return "\n".join(lines)
