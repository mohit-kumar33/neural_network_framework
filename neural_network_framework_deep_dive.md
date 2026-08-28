# neural_network_framework — Complete Deep-Dive Teaching Guide

> **Who this is for:** Anyone who wants to understand exactly how this framework works —
> every file, every function, every line, and every design decision — without needing to
> look anything up elsewhere. Every Python trick, every piece of math, every "why" is
> explained here from first principles.

---

## Table of Contents

1. [Before We Begin — Concepts You Must Know](#1-before-we-begin)
2. [The Big Picture — How neural_network_framework Works](#2-the-big-picture)
3. [tensor.py — The Autograd Engine](#3-tensorpy)
4. [nn/module.py — The Module Base Class](#4-nnmodulepy)
5. [nn/layers.py — Trainable Layers](#5-nnlayerspy)
6. [nn/activations.py — Activation Functions](#6-nnactivationspy)
7. [loss.py — Loss Functions](#7-losspy)
8. [optim/base.py — Optimizer Base Class](#8-optimbasepy)
9. [optim/sgd.py — SGD Optimizer](#9-optimsgdpy)
10. [optim/adam.py — Adam & AdamW](#10-optimadampy)
11. [utils/init.py — Weight Initializers](#11-utilsinitpy)
12. [utils/data.py — Dataset & DataLoader](#12-utilsdatapy)
13. [viz.py — Computation Graph Visualizer](#13-vizpy)
14. [A Complete Training Step — Everything Together](#14-full-training-step)
15. [Python Mechanics Master Reference](#15-python-mechanics-reference)
16. [Math Reference](#16-math-reference)

---

## 1. Before We Begin

This section explains every foundational concept used throughout the codebase.
If you already know something, skip it. But if anything in the code looks mysterious,
the answer is almost certainly here.

---

### What is a Tensor?

A **tensor** is simply a multi-dimensional array of numbers.

- A single number like `3.14` is a **scalar** — a 0-dimensional tensor.
- A list like `[1, 2, 3]` is a **vector** — a 1-dimensional tensor, shape `(3,)`.
- A table like `[[1,2],[3,4]]` is a **matrix** — a 2-dimensional tensor, shape `(2,2)`.
- A stack of matrices is a **3D tensor**, shape `(batch, rows, cols)`.

In neural_network_framework, everything — inputs, weights, biases, activations, losses — is a `Tensor` object.
`Tensor` wraps a NumPy array and adds automatic gradient tracking on top of it.

---

### What is a Gradient?

A gradient is the answer to the question: *"if I change this number slightly, how much
does the output (the loss) change?"*

Example: if `f(x) = x²`, then at `x=3` the gradient is `∂f/∂x = 2x = 6`.
This means: increasing `x` by `0.001` increases `f` by approximately `6 × 0.001 = 0.006`.

In a neural network, the loss (a single number measuring how wrong the model is) depends
on every weight through a long chain of operations. The gradient of the loss with respect
to each weight tells us which direction to nudge that weight to reduce the loss.

```
loss goes DOWN when weights move in the OPPOSITE direction of the gradient.
```

That is gradient descent:  `weight = weight − learning_rate × gradient`

---

### What is the Chain Rule?

Neural networks are composed functions: `loss = f(g(h(x)))`.

The **chain rule** from calculus says:

```
∂loss/∂x = (∂loss/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

In words: to find how `x` affects the loss, multiply all the local derivatives along the path.

This is the entire mathematical foundation for backpropagation. neural_network_framework applies the
chain rule **automatically** by attaching a backward function to every operation.

---

### Python Concepts Used in neural_network_framework

#### Closures

A **closure** is a function that remembers variables from the scope where it was defined,
even after that scope has returned.

```python
def make_adder(n):
    def add(x):
        return x + n    # n is "closed over" — still accessible after make_adder returns
    return add

add5 = make_adder(5)
print(add5(3))   # prints 8
```

Every `_backward` function in neural_network_framework is a closure. When you compute `c = a + b`,
a closure is created that captures `a`, `b`, and `c` and knows exactly how to compute
and accumulate their gradients.

#### Decorators

A decorator is syntax sugar for wrapping a function with another function.

`@property` makes a method behave like a read-only attribute (no parentheses needed):

```python
class Tensor:
    @property
    def shape(self):
        return self.data.shape

t = Tensor([1,2,3])
print(t.shape)    # (3,) — looks like attribute access, is actually a function call
```

`@classmethod` receives the class `cls` as its first argument instead of an instance `self`.
Used for factory methods that create new objects:

```python
@classmethod
def zeros(cls, *shape):
    return cls(np.zeros(shape))   # cls is the Tensor class itself
```

#### Generators and `yield`

A generator is a function that produces values one at a time using `yield`.
Python pauses the function at each `yield` and resumes when the next value is needed.

```python
def count_up(n):
    i = 0
    while i < n:
        yield i     # pause here, send i to caller
        i += 1      # resume here next time

for x in count_up(3):
    print(x)   # prints 0, 1, 2 — lazily, one at a time
```

`DataLoader.__iter__` is a generator — it yields one batch at a time, so the entire
dataset never needs to be in memory at once.

#### Dunder Methods (Magic Methods)

Python calls these automatically in response to operations:

| Method | Triggered by |
|--------|-------------|
| `__init__` | `MyClass(...)` |
| `__call__` | `obj(...)` — makes objects callable like functions |
| `__add__` | `a + b` |
| `__mul__` | `a * b` |
| `__matmul__` | `a @ b` |
| `__repr__` | `print(obj)` or `repr(obj)` |
| `__len__` | `len(obj)` |
| `__getitem__` | `obj[i]` |
| `__iter__` | `for x in obj:` |

#### `id(obj)` — Object Identity

Every Python object has a unique integer ID (its memory address):

```python
a = [1, 2, 3]
b = a           # b points to the same list
c = [1, 2, 3]   # c is a different list, same content

id(a) == id(b)  # True  — same object in memory
id(a) == id(c)  # False — different objects
```

neural_network_framework uses `id()` extensively to deduplicate tensors: if the same weight tensor is used
twice, its `id()` ensures it is only visited once during graph traversal.

#### `vars(obj)` / `obj.__dict__`

Returns a dictionary of all instance attributes:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

fido = Dog("Fido", 3)
vars(fido)   # → {'name': 'Fido', 'age': 3}
```

`Module.parameters()` uses `vars()` to automatically discover all tensors stored as
attributes — you never need to register them manually.

#### `set` — Unordered Collection of Unique Items

```python
s = set()
s.add(1)
s.add(2)
s.add(1)   # duplicate — silently ignored
print(s)   # {1, 2}
```

`tensor._prev` is a `set` so that if the same parent tensor is referenced twice
(e.g., `c = a + a`), it is only visited once during backward.

#### `defaultdict` and `deque`

`defaultdict(list)` is a dict that automatically creates an empty list for any missing key —
no need to check `if key not in d: d[key] = []` each time.

`deque` (double-ended queue) is optimised for `append` and `popleft` — both O(1).
This makes it the correct data structure for BFS (breadth-first search), where you
add to the back and remove from the front.

#### Type Hints

```python
def _collect(root: Tensor) -> Tuple[List[Tensor], List[Tuple[int, int]]]:
```

Type annotations describe what types a function accepts and returns. They have **zero**
effect on runtime behaviour. They exist purely to help you and your IDE understand the code.
`Tuple[List[Tensor], List[Tuple[int, int]]]` means: "returns a tuple whose first element
is a list of Tensors and second is a list of (int, int) pairs."

#### `from __future__ import annotations`

Without this, writing `-> "Tensor"` inside the `Tensor` class would fail because
Python tries to evaluate the name `Tensor` before the class is fully defined.
This import makes all annotations lazy (evaluated as strings only when needed).
Always the first line in files with self-referential type hints.

---

## 2. The Big Picture

Here is exactly what happens when you train a neural network with neural_network_framework:

```
 for X_batch, y_batch in dataloader:
     ①  logits = model(X_batch)       ← forward pass builds the computation graph
     ②  loss = criterion(logits, y)   ← compute scalar loss
     ③  model.zero_grad()             ← clear gradients from previous step
     ④  loss.backward()               ← backward pass fills .grad on every parameter
     ⑤  optimizer.step()              ← update parameters using .grad
```

**Step ①: Forward pass**
Every operation (`@`, `+`, `relu`, etc.) computes its output AND attaches a backward closure.
A computation graph grows in memory, connected via `._prev` links.

**Step ②: Loss**
The loss function computes a single scalar — "how wrong is the model?"

**Step ③: Clear gradients**
`model.zero_grad()` resets every parameter's `.grad` to zero. Must happen BEFORE backward.

**Step ④: Backward pass**
`loss.backward()` starts at the loss (gradient = 1.0) and walks the graph backward,
calling each closure in topological order. After this, every parameter has
`∂loss/∂param` stored in `.grad`.

**Step ⑤: Update weights**
`optimizer.step()` reads `.grad` on every parameter and applies the update rule.

---

## 3. `tensor.py` — The Autograd Engine

**File:** `neural_network_framework/tensor.py`
This is the foundation. Every other file depends on it.

---

### Module-level imports

```python
from __future__ import annotations
```
Makes all type annotations lazy. Required because `Tensor` methods reference `Tensor` itself.

```python
import numpy as np
```
All actual array math is delegated to NumPy. neural_network_framework never loops over individual elements —
NumPy's C-level routines are orders of magnitude faster.

```python
from typing import Optional, Tuple, Set
```
`Optional[X]` means "either X or None". These are documentation-only at runtime.

---

### `_ensure_tensor(x)`

```python
def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)
```

Called at the top of every binary operation. When you write `tensor + 3.14`, Python calls
`tensor.__add__(3.14)`. The `3.14` is a raw float, not a Tensor. This function wraps it
so all subsequent code can treat both operands identically.

`isinstance(x, Tensor)` returns True if `x` is a Tensor or any subclass of Tensor.
If it is already a Tensor, it is returned as-is — no copying, no overhead.

---

### `_unbroadcast(grad, target_shape)`

This is one of the most important helpers, and the subtlest.

**The problem:** NumPy allows broadcasting. You can add a `(4,)` array to a `(32, 4)` array —
NumPy silently repeats the `(4,)` array 32 times. This works for the forward pass.
But the gradient coming back has shape `(32, 4)` and must be reduced to `(4,)` by summing
over the broadcast axis.

```python
if grad.shape == target_shape:
    return grad
```
If shapes already match, no reduction needed.

```python
ndim_diff = grad.ndim - len(target_shape)
padded = (1,) * ndim_diff + target_shape
```
Left-pad the target shape with `1`s to match `grad`'s number of dimensions.

**Example:** `grad.shape=(3, 4)`, `target_shape=(4,)` → `ndim_diff=1`, `padded=(1, 4)`

```python
sum_axes = tuple(i for i, s in enumerate(padded) if s == 1)
if sum_axes:
    grad = grad.sum(axis=sum_axes, keepdims=True)
```
Sum over every axis that was `1` in the padded target (i.e., broadcast axes).
`keepdims=True` preserves the number of dimensions, needed for the reshape.

```python
if ndim_diff > 0:
    grad = grad.squeeze(axis=tuple(range(ndim_diff)))
return grad.reshape(target_shape)
```
Remove leading dummy dimensions, then reshape to exactly the target shape.

---

### `class Tensor` — Constructor `__init__`

```python
def __init__(self, data, requires_grad=False, _children=(), _op="", dtype=np.float32):
```

Every parameter explained:

- **`data`** — The actual numbers. Accepts Python lists, NumPy arrays, scalars, or another
  Tensor. Always normalised to a `float32` NumPy array internally.
- **`requires_grad`** — If True, gradients are computed for this tensor. Model weights have
  this set to True. Raw input data usually does not.
- **`_children`** — A tuple of the tensors this was computed from. When you write `c = a + b`,
  `c` is created with `_children=(a, b)`. This records the edges of the computation graph.
  The underscore prefix is a Python convention meaning "private".
- **`_op`** — A human-readable string like `"Add"`, `"MatMul"`, `"ReLU"`. Only for debugging
  and the visualizer. Zero effect on computation.
- **`dtype`** — NumPy data type. Always `float32` (32-bit floating point). PyTorch also defaults
  to float32 for performance reasons — float64 is more precise but uses twice the memory.

```python
if isinstance(data, Tensor):
    data = data.data
```
Unwrap a Tensor passed as data. Prevents Tensors being nested inside Tensors.

```python
if isinstance(data, np.ndarray):
    self.data = data.astype(dtype)
else:
    self.data = np.array(data, dtype=dtype)
```
No matter what you pass in, it always ends up as a float32 NumPy array in `self.data`.
`np.array(...)` handles Python lists, tuples, and scalars automatically.

```python
self.grad: Optional[np.ndarray] = None
```
Gradient starts as `None`. It only becomes a NumPy array when `backward()` reaches this tensor.
This is "lazy allocation" — input data tensors that never need gradients waste no memory.

```python
self._backward = lambda: None
```
The default backward function does nothing. This represents leaf nodes (raw data or parameters).
When the backward walk reaches a leaf, calling `_backward()` correctly does nothing —
because there are no parent tensors to push gradients into.

`lambda: None` is an anonymous function that takes no arguments and returns None.

```python
self._prev: Set[Tensor] = set(_children)
```
Parent tensors stored as a `set`. A set deduplicates automatically: `c = a + a` creates
`_children=(a, a)`, but `set((a, a)) = {a}` — `a` is only visited once during backward.

```python
self._op: str = _op
```
Stores the op name string for debugging and visualization.

---

### Properties

```python
@property
def shape(self) -> Tuple[int, ...]:
    return self.data.shape
```
`@property` makes `tensor.shape` look like attribute access even though it's a method call.
Delegates to NumPy.

```python
@property
def T(self) -> "Tensor":
    return self.transpose()
```
Lets you write `W.T` instead of `W.transpose()`, same as NumPy and PyTorch.

---

### `__repr__`

```python
def __repr__(self) -> str:
    grad_info = f", grad_fn=<{self._op}>" if self._op else ""
    return f"Tensor({self.data}{grad_info})"
```
Controls `print(tensor)`. If the tensor was created by an operation, shows
`grad_fn=<Add>` etc., mimicking PyTorch's style.

---

### `zero_grad`

```python
def zero_grad(self):
    if self.grad is not None:
        self.grad.fill(0.0)
    else:
        self.grad = np.zeros_like(self.data)
```
Resets gradient to zero. Called before every backward pass to prevent accumulation
across training steps.

**Why `.fill(0.0)` instead of `self.grad = np.zeros_like(...)`?**
`.fill(0.0)` mutates the existing array in-place — no new memory allocation.
Re-using the same memory is faster and reduces garbage collector pressure.

---

### `_init_grad`

```python
def _init_grad(self):
    if self.grad is None:
        self.grad = np.zeros_like(self.data)
```
Private helper called inside `_backward` closures before `self.grad += something`.
Creates the gradient array if it does not exist yet. Unlike `zero_grad()`, this never
resets an already-existing gradient — it only creates it if missing.

---

### `backward` — The Engine

```python
def backward(self, grad: Optional[np.ndarray] = None):
```

This triggers the entire reverse-mode autodiff sweep.

```python
    if not self.requires_grad:
        return
```
If this tensor does not need gradients, skip immediately.

```python
    if grad is None:
        if self.data.size != 1:
            raise RuntimeError("backward() called on non-scalar without a gradient.")
        grad = np.ones_like(self.data)
```
When you call `loss.backward()` with no argument, `loss` must be a scalar.
The gradient of a scalar with respect to itself is exactly 1.0.

If you call backward on a non-scalar without providing a gradient, it's mathematically
ambiguous — so we raise an error.

```python
    self._init_grad()
    self.grad += grad
```
Set the starting gradient for the root node (the loss) to 1.0.

```python
    topo: list[Tensor] = []
    visited: set[Tensor] = set()

    def build_topo(v: Tensor):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)
```
Depth-first search starting from the loss, recursively visiting all parents.
A node is appended to `topo` only AFTER all its parents are visited.
Result: topological order where the loss is last and leaves are first.

**Why topological order?** When `node._backward()` runs, `node.grad` must already be
fully accumulated from all downstream nodes. Reverse topological order guarantees this.

```python
    for node in reversed(topo):
        node._backward()
```
Walk from loss back to leaves, calling each backward closure.
After this loop, every parameter's `.grad` holds `∂loss/∂param`.

---

### `__add__`

```python
def __add__(self, other):
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
```

**The math:** If `out = self + other`, then `∂out/∂self = 1` and `∂out/∂other = 1`.
By chain rule: `∂loss/∂self = ∂loss/∂out × 1 = out.grad`.
Both parents receive `out.grad` directly.

**`requires_grad=self.requires_grad or other.requires_grad`:** If either input needs grad,
the output does too — gradients must flow through the entire connected subgraph.

**`+=` not `=`:** We accumulate gradients, never overwrite.
A tensor can participate in multiple operations (e.g., `W` used in both `x @ W` and `reg = W**2`).
Each usage contributes a gradient that must be summed.

**`_unbroadcast`:** Handles the case where NumPy broadcast a smaller tensor. The gradient must
be summed back to the original shape.

---

### `__radd__`

```python
def __radd__(self, other) -> "Tensor":
    return self.__add__(other)
```
Python calls `__radd__` when the left operand fails to handle the operation.
`1 + tensor` would fail because `int.__add__` does not know about Tensors.
With `__radd__`, Python falls back to `tensor.__radd__(1)` → `tensor.__add__(1)`.
Addition is commutative, so this is correct.

---

### `__sub__`

**The math:** `out = self - other` → `∂out/∂self = +1`, `∂out/∂other = -1`.
That is why the gradient flowing into `other` is negated: `other.grad += -out.grad`.

---

### `__mul__`

```python
    def _backward():
        if self.requires_grad:
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
        if other.requires_grad:
            other.grad += _unbroadcast(out.grad * self.data, other.shape)
```

**The math:** `out = self × other` → `∂out/∂self = other`, `∂out/∂other = self`.
Chain rule: `∂loss/∂self = out.grad × other.data`.

Each operand's gradient depends on the OTHER operand's value.
The closure captures `other.data` and `self.data` from the enclosing scope at the time
the forward op ran — this is the closure mechanism at work.

---

### `__truediv__`

**The math:** `out = self / other` →
- `∂out/∂self = 1/other`
- `∂out/∂other = -self / other²`

Chain rule: `∂loss/∂other = out.grad × (-self / other²)`.

---

### `__pow__`

```python
    self.grad += out.grad * exponent * (self.data ** (exponent - 1))
```

**The math:** Power rule — `∂(xⁿ)/∂x = n × x^(n-1)`.

```python
assert isinstance(exponent, (int, float))
```
Only scalar exponents supported. Tensor exponents would require a completely different
formula involving `log`.

---

### `__neg__`

```python
def __neg__(self):
    return self * -1
```
Reuses `__mul__`. The gradient of `-x` w.r.t. `x` is `-1`, which `__mul__` computes
automatically. A key design principle: compose from existing ops rather than writing
separate backward code.

---

### `__matmul__`

```python
    def _backward():
        if self.requires_grad:
            g = out.grad @ other.data.swapaxes(-1, -2)   # dL/dA = dL/dC @ B^T
            self.grad += _unbroadcast(g, self.shape)
        if other.requires_grad:
            g = self.data.swapaxes(-1, -2) @ out.grad    # dL/dB = A^T @ dL/dC
            other.grad += _unbroadcast(g, other.shape)
```

**The math:** For `C = A @ B` where shapes are `A=(m,n)`, `B=(n,p)`, `C=(m,p)`:
- `∂loss/∂A = ∂loss/∂C @ B^T`  shapes: `(m,p) @ (p,n) = (m,n)` ✓
- `∂loss/∂B = A^T @ ∂loss/∂C`  shapes: `(n,m) @ (m,p) = (n,p)` ✓

`swapaxes(-1, -2)` transposes the last two axes. For a 2D matrix this is a full transpose.
For a batched 3D tensor `(batch, m, n)` it transposes only the last two dims: `(batch, n, m)`.
One line handles both the 2D and batched cases without special-casing.

---

### `sum`, `mean`, `max`

**`sum`:** `∂(Σxᵢ)/∂xᵢ = 1` for every element. The gradient is broadcast back to the
original shape.

```python
    def _backward():
        grad = out.grad
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis=axis)   # restore the summed axis
        self.grad += np.broadcast_to(grad, self.shape)
```

`np.expand_dims` adds back the axis that `sum` removed.
`np.broadcast_to` efficiently spreads the gradient to every element — no data copy,
just a view with broadcast strides.

**`mean`:** Same as sum but divide by `n` (the count). Each element contributed `1/n`
to the output, so each receives `out.grad / n`.

**`max`:** The gradient flows only to the maximum element(s). For ties, the gradient
is split equally. `mask /= mask.sum(...) + 1e-10` normalises ties;
`1e-10` prevents division by zero.

---

### `exp`, `log`

**`exp`:**
```python
val = np.exp(self.data)
...
self.grad += out.grad * val   # ∂eˣ/∂x = eˣ
```
The derivative of `exp` is itself — the most beautiful property in calculus.
`val` is cached in the closure so we don't recompute `np.exp` during backward.

**`log`:**
```python
np.log(np.clip(self.data, 1e-7, None))
...
self.grad += out.grad / np.clip(self.data, 1e-7, None)   # ∂log(x)/∂x = 1/x
```
`np.clip(..., 1e-7, None)` prevents `log(0) = -infinity` and division by zero.
The same clip is applied in both forward and backward for consistency.

---

### `reshape`, `transpose`

```python
# reshape backward — just inverse reshape
self.grad += out.grad.reshape(self.shape)

# transpose backward — inverse permutation
inv = np.argsort(axes)
self.grad += out.grad.transpose(inv)
```
Reshape does not change values, only layout. The backward is the inverse reshape.

For transpose: if `axes=(2,0,1)`, then `np.argsort([2,0,1]) = [1,2,0]` computes the
permutation that exactly reverses the original — element 0 came from position 1, etc.

---

### Factory Class Methods

```python
@classmethod
def zeros(cls, *shape, dtype=np.float32, requires_grad=False) -> "Tensor":
    return cls(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)
```

`@classmethod` — first argument is the class itself (`cls`), not an instance.
`*shape` — accepts any number of positional args collected into a tuple:
`Tensor.zeros(3, 4)` → `shape=(3, 4)`.

Same pattern for `.ones()`, `.randn()`, `.from_numpy()`.

---

## 4. `nn/module.py` — The Module Base Class

**File:** `neural_network_framework/nn/module.py`
Every layer, activation, and loss function inherits from `Module`.

---

### `__call__`

```python
def __call__(self, *args, **kwargs) -> Tensor:
    return self.forward(*args, **kwargs)
```

`__call__` is invoked when you write `model(x)`. It routes to `forward`, which subclasses
must implement. `*args` captures positional arguments; `**kwargs` captures keyword arguments.
Both are forwarded unchanged.

This is the same pattern as PyTorch's `nn.Module`.

---

### `forward`

```python
def forward(self, *args, **kwargs) -> Tensor:
    raise NotImplementedError(
        f"{self.__class__.__name__} must implement forward()"
    )
```
Abstract method. Every subclass must override it. `self.__class__.__name__` is the
subclass's name (e.g., `"Linear"`), giving a clear error message.

---

### `parameters`

```python
def parameters(self) -> list[Tensor]:
    params = []
    seen: set[int] = set()

    def _collect(obj):
        for attr_val in vars(obj).values():
            if isinstance(attr_val, Tensor) and attr_val.requires_grad:
                if id(attr_val) not in seen:
                    seen.add(id(attr_val))
                    params.append(attr_val)
            elif isinstance(attr_val, Module):
                _collect(attr_val)
            elif isinstance(attr_val, (list, tuple)):
                for item in attr_val:
                    if isinstance(item, Module):
                        _collect(item)

    _collect(self)
    return params
```

Recursive traversal of the entire object graph:
- Tensor with `requires_grad=True` → parameter, collect it
- Another Module → recurse (finds nested layers)
- list/tuple → check each element for Modules

`seen` deduplicates by `id()` — the same tensor referenced multiple times is collected once.

**The magic:** You never register parameters manually. Just assign:
`self.weight = Tensor(..., requires_grad=True)` in `__init__` and it's auto-discovered.

---

### `zero_grad`

```python
def zero_grad(self):
    for p in self.parameters():
        p.zero_grad()
```
Clears all parameter gradients. Must be called before every `loss.backward()`.

---

### `train` and `eval`

```python
def train(self, mode: bool = True):
    self.training = mode
    for attr_val in vars(self).values():
        if isinstance(attr_val, Module):
            attr_val.train(mode)   # propagate to sub-modules
    return self

def eval(self):
    return self.train(False)
```

Recursively sets `self.training` on the module and all sub-modules.
Layers like `Dropout` and `BatchNorm1d` check `self.training` in their forward to
change behaviour between training and inference.

---

### `__init_subclass__` — Automatic Training Mode

```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        self.training = True          # always set first
        original_init(self, *args, **kwargs)

    cls.__init__ = patched_init
```

A Python metaclass hook: called automatically whenever a class inherits from `Module`.

It patches every subclass's `__init__` to **always** set `self.training = True` first,
before the subclass's own `__init__` runs.

**Why is this clever?** Without this, every layer would need to explicitly call
`super().__init__()`. Developers often forget this. The hook makes it invisible —
every Module automatically starts in training mode.

---

### `class Sequential`

```python
class Sequential(Module):
    def __init__(self, *modules: Module):
        self.layers = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

Takes any number of modules and stores them in `self.layers`.
The forward pass runs each layer in order, feeding the output of one into the next.
This is the definition of a feedforward sequential network.

---

## 5. `nn/layers.py` — Trainable Layers

**File:** `neural_network_framework/nn/layers.py`

---

### `class Linear`

The most fundamental building block: computes `y = x @ W^T + b`.

```python
def __init__(self, in_features, out_features, bias=True, init="he"):
    w_data = initialize_weights(in_features, out_features, method=init)
    self.weight = Tensor(w_data, requires_grad=True)
    if bias:
        self.bias = Tensor(np.zeros(out_features, ...), requires_grad=True)
```

**Weight shape is `(out_features, in_features)`** — stored transposed relative to math notation.
This lets `x @ W.T` produce shape `(batch, out_features)` naturally.

**Bias initialised to zeros:** The weight initializer already breaks symmetry.
No benefit to a nonzero starting bias.

**`requires_grad=True`:** These are learnable parameters. The autograd engine computes
gradients for them during backward.

```python
def forward(self, x: Tensor) -> Tensor:
    out = x @ self.weight.T
    if self.bias is not None:
        out = out + self.bias
    return out
```

If `x=(N, in_f)` and `self.weight=(out_f, in_f)`, then `self.weight.T=(in_f, out_f)`,
and `x @ W.T = (N, out_f)`. The bias `(out_f,)` is broadcast across the batch dimension.

Every `@` and `+` here goes through `Tensor.__matmul__` and `Tensor.__add__` which
build the autograd graph automatically. No manual gradient code in `Linear.forward`.

---

### `class BatchNorm1d`

Batch Normalization (Ioffe & Szegedy, 2015) normalizes activations across a mini-batch.
This stabilizes training and allows much higher learning rates.

```python
self.gamma = Tensor(np.ones(num_features, ...), requires_grad=True)
self.beta  = Tensor(np.zeros(num_features, ...), requires_grad=True)
self.running_mean = np.zeros(num_features, ...)
self.running_var  = np.ones(num_features, ...)
```

- `gamma` (scale) and `beta` (shift) are learnable. They let the network undo normalization
  if it helps — the layer can learn to be an identity function.
- `running_mean` and `running_var` are plain NumPy arrays (not Tensors). They track
  exponential moving averages of batch statistics, used only during eval mode.

**Forward pass:**

```python
    if self.training:
        mu  = x.data.mean(axis=0)
        var = x.data.var(axis=0)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
        self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
    else:
        mu  = self.running_mean
        var = self.running_var
```

Training: compute statistics from the current batch. Update the running stats as an
exponential moving average (EMA). `momentum=0.1` means 10% new + 90% old.

Eval: use the stored running stats. This makes eval deterministic.

```python
    inv_std    = 1.0 / np.sqrt(var + self.eps)
    x_hat_data = (x.data - mu) * inv_std
```

Normalize: subtract mean, divide by standard deviation.
`eps=1e-5` inside the sqrt prevents division by zero when variance is zero.

**Why cache `_mu`, `_inv_std`, etc.?**

```python
    _mu = mu; _inv_std = inv_std; _x_hat = x_hat_data; _N = x.data.shape[0]
```

Python closures capture variables by name (by reference), not by value.
If `mu` were reassigned on the next forward call, the backward closure from THIS call
would see the new value. Assigning `_mu = mu` creates a new local variable that won't
be overwritten — it's permanently bound to the value from this particular forward pass.

**Backward pass:**

```python
    def _backward():
        dout = out.grad   # (N, C)
        dx_hat = dout
        dvar = (-0.5 * (dx_hat * _x_hat).sum(axis=0) * _inv_std ** 2)
        dmu  = (-dx_hat * _inv_std).sum(axis=0) + dvar * (-2.0 / _N) * (x.data - _mu)
        dx   = (dx_hat * _inv_std
                + dvar * 2.0 * (x.data - _mu) / _N
                + dmu / _N)
```

This is the full BatchNorm backward from the original paper. The three terms of `dx` come
from three paths through which `x` affects the output:
1. Direct path through `x_hat = (x - mu) * inv_std`
2. Indirect path through `var` (which depends on `x`)
3. Indirect path through `mu` (which depends on `x`)

After normalization, the learned affine transform is applied:
```python
    if self.affine and self.gamma is not None:
        out = self.gamma * out + self.beta
```
Because `gamma` and `beta` are Tensors with `requires_grad=True`, this multiplication and
addition automatically build autograd graph nodes. Their gradients are computed for free.

---

### `class Dropout`

```python
def forward(self, x: Tensor) -> Tensor:
    if not self.training or self.p == 0.0:
        return x
```
During eval (or when `p=0`), return the input unchanged.

```python
    scale = 1.0 / (1.0 - self.p)
    mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) * scale
```

`np.random.rand(*x.shape)` generates uniform random values in `[0,1)` with the same shape as `x`.
Elements greater than `self.p` survive. Survivors are scaled by `1/(1-p)`.

**Why scale?** Without scaling, dropout during training would reduce expected activation by `(1-p)`.
This mismatch would mean eval mode (no dropout) sees different magnitude inputs.
Scaling by `1/(1-p)` during training keeps expected values the same, so eval is simply `return x`.
This is called **inverted dropout** — the standard in modern frameworks.

```python
    def _backward():
        x.grad += out.grad * mask
```
Same mask used in forward is reused in backward. Zeroed neurons pass zero gradient —
they contributed nothing, so they receive no gradient update.

---

### `class Embedding`

```python
self.weight = Tensor(
    np.random.randn(num_embeddings, embedding_dim) * 0.02,
    requires_grad=True,
)
```
A lookup table of shape `(vocab_size, embed_dim)`.
Initialised with small normal values so embeddings start near zero.

```python
def forward(self, indices) -> Tensor:
    idx = np.array(indices).flatten().astype(int)
    selected = self.weight.data[idx]   # fancy indexing: select rows by index

    def _backward():
        np.add.at(self.weight.grad, idx, out.grad)
```

`self.weight.data[idx]` is NumPy **fancy indexing** — selects specific rows.
For `idx = [2, 5, 2]`, it returns rows 2, 5, 2 of the embedding table (row 2 twice).

**Why `np.add.at` instead of `self.weight.grad[idx] += out.grad`?**

If `idx = [2, 2]` (the same word appears twice in a batch):
```python
# WRONG: NumPy buffers += with fancy indexing
self.weight.grad[[2,2]] += grad   # second += is silently ignored
# result: row 2 gets only one gradient, not two

# CORRECT: np.add.at is unbuffered
np.add.at(self.weight.grad, [2, 2], grad)  # correctly accumulates both gradients
```
This is a well-known NumPy gotcha. `np.add.at` is the correct tool for scatter-add
with repeated indices.

---

## 6. `nn/activations.py` — Activation Functions

**File:** `neural_network_framework/nn/activations.py`
All activations are stateless Modules — no learnable parameters.

---

### `ReLU`

```
f(x) = max(0, x)
∂f/∂x = 1 if x > 0, else 0
```

```python
mask = (x.data > 0).astype(np.float32)
out = Tensor(x.data * mask, ...)

def _backward():
    x.grad += out.grad * mask
```

`mask` is a binary array: 1.0 where `x > 0`, 0.0 elsewhere.
Forward: multiply by mask zeros out negative values.
Backward: the same mask zeros out gradients for deactivated positions.

---

### `LeakyReLU`

```
f(x) = x if x > 0, else slope × x
∂f/∂x = 1 if x > 0, else slope
```

Prevents "dead neurons" — neurons that get permanently stuck at zero gradient
and can never recover their weights.

---

### `Sigmoid`

```
f(x) = 1 / (1 + e^(-x))
∂f/∂x = f(x) × (1 - f(x))
```

```python
sig = 1.0 / (1.0 + np.exp(-x.data.clip(-500, 500)))
...
def _backward():
    x.grad += out.grad * sig * (1.0 - sig)
```

The derivative is expressed entirely in terms of the output — no need to store the input.
`clip(-500, 500)` prevents `np.exp(1000) = inf` overflow.

---

### `Tanh`

```
f(x) = tanh(x)
∂f/∂x = 1 - tanh(x)²
```

Same elegant property as Sigmoid: derivative expressed in terms of the output value.

---

### `GELU`

Used in BERT, GPT-2, and most modern transformers. Approximates `x × Φ(x)` where
`Φ` is the standard normal CDF.

```python
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_COEFF = 0.044715

inner    = _SQRT_2_OVER_PI * (x + _COEFF * x**3)
tanh_val = np.tanh(inner)
val      = 0.5 * x * (1.0 + tanh_val)
```

The constants `sqrt(2/π)` and `0.044715` come from the GELU paper (Hendrycks & Gimpel, 2016).
This is an approximation to the exact CDF that is computationally efficient.

The backward applies the product rule on `0.5 × x × (1 + tanh(inner))`.

---

### `Softmax`

```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

```python
shifted = x.data - x.data.max(axis=self.axis, keepdims=True)
exp_x   = np.exp(shifted)
sm      = exp_x / exp_x.sum(axis=self.axis, keepdims=True)
```

**Why subtract the max?** Numerical stability. `np.exp(1000) = inf`.
Subtracting `max(x)` keeps all exponents non-positive, so they stay in `[0, 1]`.
The result is mathematically identical:
`exp(x-c) / Σ exp(xⱼ-c) = exp(x) / Σ exp(xⱼ)` (constant cancels).

```python
def _backward():
    dot = (out.grad * sm).sum(axis=self.axis, keepdims=True)
    x.grad += sm * (out.grad - dot)
```

The full Softmax Jacobian is `∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ - sⱼ)`.
This vectorized formula avoids the full n×n Jacobian matrix and requires only one dot product per row.

---

## 7. `loss.py` — Loss Functions

**File:** `neural_network_framework/loss.py`

---

### `MSELoss`

```python
def forward(self, pred, target):
    diff = pred - target
    sq   = diff ** 2
    return sq.mean() if self.reduction == "mean" else sq.sum()
```

Composed entirely from Tensor operations (`-`, `**`, `.mean()`).
No manual backward needed — the autograd graph handles it through each op's backward closure.

---

### `BCELoss`

```
L = -mean(y × log(p) + (1-y) × log(1-p))
∂L/∂p = -y/p + (1-y)/(1-p)
```

```python
p = pred.data.clip(eps, 1 - eps)   # prevent log(0)
```

Custom backward is written (rather than composing from primitives) because it is more
efficient and numerically stable.

---

### `BCEWithLogitsLoss` — Numerically Stable BCE

Instead of `BCE(sigmoid(x), y)` (which risks overflow at large `x`),
we algebraically simplify to a single stable formula:

```
L = max(x, 0) - x×y + log(1 + e^(-|x|))
```

```python
loss_data = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
```

`np.log1p(u)` computes `log(1 + u)` with better precision for small `u`.

```python
def _backward():
    sig  = 1.0 / (1.0 + np.exp(-x))
    grad = sig - y    # elegant: sigmoid(logit) minus target
```

The gradient simplifies to `sigmoid(logit) - target`. This is the logistic regression
gradient — simple, numerically clean.

---

### `CrossEntropyLoss`

Combines log-softmax and NLL in one numerically stable pass.

```python
shifted     = logits.data - logits.data.max(axis=1, keepdims=True)  # stability
log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
log_probs   = shifted - log_sum_exp                                   # log-softmax
nll         = -log_probs[np.arange(N), y]                            # select correct class
```

`np.arange(N) = [0, 1, ..., N-1]` paired with `y = [class_0, class_1, ...]`
selects one log-probability per sample (the correct class's log-prob).
This is NumPy advanced integer indexing.

```python
def _backward():
    sm   = np.exp(log_probs)          # softmax recovered from log-softmax
    grad = sm.copy()
    grad[np.arange(N), y] -= 1.0     # subtract 1 at correct class positions
    grad /= N                          # account for mean reduction
    logits.grad += out.grad * grad
```

**The cleanest gradient in deep learning:** `softmax(logits) - one_hot(target)`.
Just subtract 1 from the correct class's softmax score. Everything else is unchanged.
The `log` and `exp` cancel perfectly, making this exact and numerically clean.

---

## 8. `optim/base.py` — Optimizer Base Class

```python
class Optimizer:
    def __init__(self, params: list[Tensor]):
        if not params:
            raise ValueError("Optimizer requires at least one parameter.")
        self.params = list(params)   # copy, not alias

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError
```

`list(params)` makes a copy of the parameter list. Later mutations to the original
(e.g., unfreezing parameters) won't silently affect the optimizer.

`zero_grad` is implemented here so all subclasses get it for free.
`step` is abstract — every subclass must implement its own update rule.

---

## 9. `optim/sgd.py` — SGD Optimizer

**Update rules:**

```
Plain SGD:        θ ← θ - lr × g
With momentum:    v ← μ×v + (1-dampening)×g    θ ← θ - lr × v
With Nesterov:    g ← g + μ×v                  θ ← θ - lr × g
With weight decay: add λ×θ to g before the above
```

```python
self._velocities: list[np.ndarray | None] = [None] * len(self.params)
```
Pre-allocates `None` placeholders for velocity buffers (one per parameter).
Initialised lazily — no memory allocated before the first training step.

```python
def step(self):
    for i, p in enumerate(self.params):
        if p.grad is None:
            continue
        g = p.grad.copy()   # copy! prevents in-place mutation of .grad
```

**Why `.copy()`?** Without it, subsequent operations (weight decay, momentum) would
mutate `p.grad` directly, corrupting gradient accumulation for any future use.

```python
        if self.weight_decay != 0.0:
            g += self.weight_decay * p.data
```
L2 regularization: adds `λθ` to the gradient. Mathematically equivalent to adding
`(λ/2)‖θ‖²` to the loss. Penalizes large weights and reduces overfitting.

```python
        if self.momentum != 0.0:
            if v is None:
                self._velocities[i] = g.copy()
            else:
                v = self.momentum * v + (1.0 - self.dampening) * g
                self._velocities[i] = v
```
**Momentum:** maintains a running average of past gradients ("velocity").
Instead of jumping in the raw gradient direction, move in the direction of accumulated
velocity. This dampens oscillations and accelerates progress in consistent directions.

```python
        if self.nesterov:
            g = g + self.momentum * v   # look-ahead
        else:
            g = v
```
**Standard momentum:** update direction = velocity.
**Nesterov momentum:** look-ahead correction. Converges faster in practice.

```python
        p.data -= self.lr * g
```
The actual weight update. Subtract (not add) because gradient descent minimizes the loss
by moving opposite to the gradient.

---

## 10. `optim/adam.py` — Adam & AdamW

**Adam algorithm (Kingma & Ba, 2014):**
```
m_t = β₁ × m_{t-1} + (1-β₁) × g         ← EMA of gradient (1st moment)
v_t = β₂ × v_{t-1} + (1-β₂) × g²        ← EMA of squared gradient (2nd moment)
m̂_t = m_t / (1 - β₁ᵗ)                   ← bias-corrected 1st moment
v̂_t = v_t / (1 - β₂ᵗ)                   ← bias-corrected 2nd moment
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)   ← parameter update
```

```python
self._m: list[np.ndarray | None] = [None] * n   # 1st moment (mean of gradients)
self._v: list[np.ndarray | None] = [None] * n   # 2nd moment (mean of squared gradients)
self._t: int = 0                                  # step counter
```

```python
def step(self):
    self._t += 1
    for i, p in enumerate(self.params):
        if p.grad is None:
            continue
        g = p.grad.copy()

        if self._m[i] is None:
            self._m[i] = np.zeros_like(p.data)
            self._v[i] = np.zeros_like(p.data)

        self._m[i] = b1 * self._m[i] + (1 - b1) * g
        self._v[i] = b2 * self._v[i] + (1 - b2) * (g ** 2)
```
Exponential moving averages. `b1=0.9`: 90% old + 10% new gradient.
`b2=0.999`: tracks variance much more slowly (gradients can be very noisy).

```python
        m_hat = self._m[i] / (1 - b1 ** t)
        v_hat = self._v[i] / (1 - b2 ** t)
```
**Bias correction:** At step 1, moment estimates start at zero.
Without correction, `m_1 = 0.1 × g` severely underestimates the true mean.
Dividing by `(1 - β₁ᵗ)` corrects this:
- At `t=1` with `β₁=0.9`: `m̂ = m/(1-0.9) = m/0.1 = 10m = g`. Correct!
- As `t→∞`, `β₁ᵗ→0`, so `m̂ → m`. Correction disappears naturally.

```python
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
```
Effective learning rate per parameter: `lr / sqrt(v_hat + eps)`.
- High gradient variance (noisy) → smaller effective step (cautious)
- Consistent gradients → larger effective step (confident)

`eps=1e-8` prevents division by zero.

---

### `class AdamW` — Decoupled Weight Decay

```python
p.data -= self.lr * self.weight_decay * p.data    # weight decay first
p.data -= self.lr * m_hat / (np.sqrt(v_hat) + eps)  # then Adam update
```

**Why this order matters:** In regular Adam, weight decay is added to `g` before the Adam
update. The adaptive `1/sqrt(v_hat)` term then rescales the effective weight decay
differently per parameter — the actual decay becomes `wd/sqrt(v_hat)`, which is unintended.

AdamW applies weight decay directly to the parameters as a multiplicative shrinkage,
completely separate from the gradient. Effective decay is always exactly `wd`.

AdamW is the standard optimizer for training transformers (BERT, GPT, LLaMA) for this reason.

---

## 11. `utils/init.py` — Weight Initializers

**Why initialization matters:** If all weights start at zero, all neurons compute
identical gradients and learn identically — the network never differentiates.
If weights are too large, gradients explode. Too small, gradients vanish.
Good initialization keeps activation variance roughly constant across layers.

---

### `xavier_uniform`

```python
limit = np.sqrt(6.0 / (fan_in + fan_out))
return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
```

**Derivation:** For `y = Wx` with mean-zero `W` and `x`:
`Var(yᵢ) = fan_in × Var(Wᵢⱼ) × Var(xⱼ)`

To keep variance constant (no growth or decay): `Var(W) = 2/(fan_in + fan_out)`.

For `Uniform(-a, a)`: `Var = a²/3`. Setting equal: `a = sqrt(6 / (fan_in + fan_out))`.

**Best for:** `tanh`, `sigmoid` activations.

### `he_uniform` and `he_normal`

```python
limit = np.sqrt(6.0 / fan_in)
# or: std = np.sqrt(2.0 / fan_in)
```

**Why `2/fan_in` instead of `2/(fan_in+fan_out)`?**
ReLU zeros out approximately half its inputs on average, halving the variance.
He initialization doubles the variance (uses `fan_in` instead of harmonic mean) to compensate.

**Best for:** `ReLU`, `LeakyReLU` activations.

### `orthogonal`

```python
M = np.random.randn(rows, cols)
U, _, Vt = np.linalg.svd(M, full_matrices=False)
out = U if rows >= cols else Vt
```

Generates an orthogonal matrix via Singular Value Decomposition (SVD).
Orthogonal matrices preserve norms: `‖Wx‖ = ‖x‖`.

**Why useful for RNNs?** Gradients in RNNs flow backward through many time steps.
An orthogonal recurrent weight matrix has eigenvalues of magnitude exactly 1.0 —
neither growing nor shrinking — preventing vanishing/exploding gradients.

---

## 12. `utils/data.py` — Dataset & DataLoader

**File:** `neural_network_framework/utils/data.py`

---

### `class Dataset`

```python
class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
```
Abstract interface. `__len__` is needed by `DataLoader` to know the total sample count.
`__getitem__` retrieves one sample by index.

---

### `class TensorDataset`

```python
def __init__(self, *arrays):
    self.arrays = []
    for a in arrays:
        if isinstance(a, Tensor):
            self.arrays.append(a.data)   # unwrap Tensor to plain NumPy
        else:
            self.arrays.append(np.array(a))

    sizes = [a.shape[0] for a in self.arrays]
    assert len(set(sizes)) == 1, "All arrays must have the same first dimension."
```

`len(set(sizes)) == 1` is an elegant check that all sizes are identical.
A set collapses duplicates — if all sizes are the same, the set has exactly one element.

```python
def __getitem__(self, idx):
    return tuple(a[idx] for a in self.arrays)
```
Returns `(X[idx], y[idx])` as a tuple.

---

### `class DataLoader`

```python
def __len__(self):
    n = len(self.dataset)
    if self.drop_last:
        return n // self.batch_size
    return (n + self.batch_size - 1) // self.batch_size
```

`(n + batch_size - 1) // batch_size` is **ceiling integer division**.
For `n=100, batch_size=32`: `(100+31)//32 = 131//32 = 4` batches.

```python
def __iter__(self) -> Iterator:
    n       = len(self.dataset)
    indices = np.arange(n)
    if self.shuffle:
        np.random.shuffle(indices)
```

`__iter__` is a generator. Every time you write `for batch in dataloader:`,
Python calls `__iter__` which starts the generator fresh.

`np.arange(n) = [0,1,...,n-1]`. Shuffling this index array randomizes access order
without touching the actual data. Re-shuffled every epoch because `__iter__` is
called fresh each time.

```python
    start = 0
    while start < n:
        end = start + self.batch_size
        if end > n and self.drop_last:
            break
        batch_indices = indices[start:min(end, n)]
        items = [self.dataset[int(idx)] for idx in batch_indices]

        if isinstance(items[0], tuple):
            batch = tuple(
                Tensor(np.stack([item[j] for item in items]))
                for j in range(len(items[0]))
            )
        else:
            batch = Tensor(np.stack(items))

        yield batch
        start = end
```

`yield batch` is the generator pause point. Execution halts here, hands the batch
to the `for` loop body, and resumes on the next iteration.

`np.stack(items)` stacks individual samples along a new axis 0:
`[array(784), array(784)]` → `array(batch_size, 784)`.

For `TensorDataset(X, y)`: each `item` is a tuple `(x_sample, y_sample)`.
The list comprehension collects all `x_samples` (j=0) or all `y_samples` (j=1)
and stacks them into separate Tensors.

---

## 13. `viz.py` — Computation Graph Visualizer

**File:** `neural_network_framework/viz.py`

This is the technical novelty of neural_network_framework. Unlike micrograd's scalar-only visualizer,
this works correctly for tensor operations of any shape, shows gradient norms after
backward(), automatically detects pathological gradients, and can export to Graphviz SVG.

---

### Module-level Constants

```python
_OP_CATEGORY: Dict[str, str] = {
    "Add": "arith",  "MatMul": "linalg",  "ReLU": "activation",  ...
}
```
Maps each operation's `._op` string to a visual category. Used for colour-coding and symbols.

```python
_CAT_SYMBOL: Dict[str, str] = {
    "arith":      "⊕",    # arithmetic: +, -, *, /
    "linalg":     "⊗",    # linear algebra: @, transpose, reshape
    "reduce":     "Σ",    # reductions: sum, mean, max
    "elemwise":   "ƒ",    # element-wise math: exp, log, abs
    "activation": "σ",    # activations: relu, sigmoid, tanh
    "layer":      "◈",    # layers: dropout, batchnorm, embedding
    "loss":       "ℒ",    # loss functions
    "leaf":       "✦",    # leaf tensors: parameters and raw data
}
```
Unicode symbols shown in ASCII boxes for instant visual identification.

```python
_DOT_COLOR: Dict[str, str] = {
    "arith":      "#AED6F1",   # light blue
    "linalg":     "#A9DFBF",   # light green
    "leaf_param": "#F7DC6F",   # gold  — learnable parameters
    "leaf_data":  "#D5D8DC",   # grey  — non-learnable inputs
    ...
}
```
Hex colour codes for Graphviz. Parameters are gold diamonds; data inputs are grey boxes.

---

### `_collect(root)` — Graph Traversal

```python
def _collect(root: Tensor) -> Tuple[List[Tensor], List[Tuple[int, int]]]:
    visited: Set[int] = set()
    nodes: List[Tensor] = []
    edges: List[Tuple[int, int]] = []

    queue: deque[Tensor] = deque([root])
    while queue:
        node = queue.popleft()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        nodes.append(node)
        for parent in node._prev:
            pid = id(parent)
            edges.append((pid, nid))   # forward direction: producer → consumer
            if pid not in visited:
                queue.append(parent)

    return nodes, edges
```

**Algorithm:** Breadth-First Search (BFS) starting from the root tensor.

BFS visits nodes level by level. The `deque` implements a FIFO queue:
- `queue.popleft()` — remove from the front (FIFO = BFS order)
- `queue.append()` — add to the back

`visited` is a set of integer `id()`s. We use `id()` rather than the Tensor objects themselves
because we need fast O(1) lookups and we don't want Python's equality check to trigger
any custom `__eq__` methods.

**Edge direction:** Edges are stored as `(producer_id, consumer_id)` — the forward data-flow
direction. Since `node._prev` stores the *inputs* to `node`, each element is a producer.

**Why BFS instead of DFS?**
Both work for collection. BFS was chosen because it naturally produces a sensible
display order (root first, then nearby nodes) and is iterative (no risk of Python's
recursion limit for very deep networks).

---

### `_topo_sort(nodes, edges)` — Kahn's Algorithm

```python
def _topo_sort(nodes, edges) -> List[Tensor]:
    id_to_node = {id(n): n for n in nodes}
    in_degree  = {id(n): 0 for n in nodes}   # how many producers feed into each node
    adj        = defaultdict(list)             # forward adjacency: producer → consumers

    for src, dst in edges:
        if src in in_degree and dst in in_degree:
            in_degree[dst] += 1
            adj[src].append(dst)

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result = []
    while queue:
        nid = queue.popleft()
        result.append(id_to_node[nid])
        for nbr in adj[nid]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    return result   # leaves first, root last
```

**Kahn's algorithm** produces topological order:
1. Start with all nodes that have `in_degree == 0` (leaves — no producers).
2. Remove them, decrement `in_degree` of their consumers.
3. Add any consumer whose `in_degree` reaches 0 (all its producers have been processed).
4. Repeat until no nodes remain.

Result: leaves first, root last. This is the order for the ASCII display:
inputs at the top, loss at the bottom.

---

### `_depth_from_root(root, nodes)` — Depth Computation

```python
def _depth_from_root(root, nodes) -> Dict[int, int]:
    depths = {id(root): 0}   # root is at depth 0
    queue  = deque([root])
    visited = set()

    while queue:
        node = queue.popleft()
        nid  = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        for parent in node._prev:
            pid   = id(parent)
            new_d = depths[nid] + 1
            if pid not in depths or depths[pid] < new_d:
                depths[pid] = new_d   # assign maximum depth
            if pid not in visited:
                queue.append(parent)

    ...
    return depths
```

BFS from root following `_prev` (backward in data-flow direction).
Root gets depth 0; each parent gets depth = (child's depth + 1).

**Why take the maximum depth?** In a DAG, a node can be reached via multiple paths
of different lengths. Assigning the maximum depth ensures the display is consistent —
leaf nodes (raw data and parameters) always appear at the greatest depth.

---

### Node Property Helpers

```python
def _is_leaf(t: Tensor) -> bool:
    return len(t._prev) == 0
```
A leaf has no parents — it's raw data or a model parameter, not computed from other tensors.

```python
def _is_param(t: Tensor) -> bool:
    return _is_leaf(t) and t.requires_grad
```
A parameter is a leaf that needs a gradient. This distinguishes weights from input data.

```python
def _category(t: Tensor) -> str:
    if _is_leaf(t):
        return "leaf"
    return _OP_CATEGORY.get(t._op, "unknown")
```
Look up the operation name in `_OP_CATEGORY`. Unknown ops get the `"unknown"` category.

```python
def _shape_str(t: Tensor) -> str:
    if t.data.ndim == 0 or t.data.size == 1:
        return "scalar"
    return "(" + "×".join(str(s) for s in t.shape) + ")"
```
Human-readable shape: `"(32×4)"` for shape `(32, 4)`, `"scalar"` for size-1 tensors.
`"×".join(...)` joins dimension sizes with `×` separators.

```python
def _grad_norm_str(t: Tensor) -> str:
    if t.grad is None:
        return ""
    gn = float(np.linalg.norm(t.grad))
    if np.isnan(gn):   flag = " ⚠ NaN"
    elif np.isinf(gn): flag = " ⚠ Inf"
    elif gn < 1e-7 and t.requires_grad: flag = " ⚠ vanish"
    elif gn > 1e3:     flag = " ⚠ explode"
    else:              flag = ""
    return f"‖∇‖={gn:.4f}{flag}"
```

`np.linalg.norm(grad)` computes the Frobenius norm of the gradient array:
`‖g‖ = sqrt(Σ gᵢ²)`. A single number summarizing gradient magnitude.

**Health flags:**
- `⚠ NaN` — Not-a-Number. Something went wrong: division by zero, log of negative, etc.
- `⚠ Inf` — Infinite. A catastrophic overflow occurred.
- `⚠ vanish` — Norm < 1e-7. This parameter receives almost no gradient signal.
  Common in deep networks without normalization.
- `⚠ explode` — Norm > 1e3. The update will be enormous and destabilize training.
  Common in RNNs without gradient clipping.

---

### `_render_ascii(root, nodes, edges, show_grads)` — ASCII Renderer

```python
    topo  = _topo_sort(nodes, edges)
    depths = _depth_from_root(root, nodes)
    max_depth = max(depths.values(), default=0)
```
Get topological order and depth map.

```python
    node_sid: Dict[int, str] = {id(n): f"#{i:02d}" for i, n in enumerate(topo)}
```
Assign stable short IDs: `#00`, `#01`, ... `{i:02d}` zero-pads to 2 digits.
Used to cross-reference shared nodes (a tensor that feeds multiple consumers).

```python
    consumers: Dict[int, List[Tensor]] = defaultdict(list)
    for src_id, dst_id in edges:
        if dst_id in node_by_id:
            consumers[src_id].append(node_by_id[dst_id])
```
Forward adjacency: for each producer, which tensors does it feed into?
Used to draw arrows below each node's box.

```python
    def bar(c="━"): lines.append(c * W)
```
Tiny helper: `bar()` appends a horizontal separator repeated `W=62` times.
`c * W` in Python repeats a string `W` times — `"━" * 5 = "━━━━━"`.

**Header:**
```python
    bar()
    lines.append(f"  neural_network_framework  ·  Computation Graph{root_val}")
    lines.append(f"  Nodes: {len(nodes)}   Edges: {len(edges)}   Depth: {max_depth}   Params: {n_params} ({param_elems:,} elements)")
    bar()
```
`{param_elems:,}` formats a large integer with comma separators: `1234567` → `1,234,567`.

**Drawing boxes:**

```python
    shown: Set[int] = set()
    for node in topo:
        ...
        if nid in shown:
            lines.append(f"  {sid}  ↑ (already drawn above)")
        else:
            shown.add(nid)
            inner = f" {sid}  {op_part:<18}  {shape_part:<14}{grad_part} "
            w = len(inner)
            lines.append("  ┌" + "─" * w + "┐")
            lines.append("  │" + inner    + "│")
            lines.append("  └" + "─" * w + "┘")
```

`shown` tracks drawn nodes. In a DAG, a node can appear multiple times in topological
order (once per path through it). The second time, we show a reference instead of
redrawing the box.

`{op_part:<18}` — left-align and pad to 18 characters, ensuring consistent column width.

```python
            lines.append("  ┌" + "─" * w + "┐")
            lines.append("  │" + inner    + "│")
            lines.append("  └" + "─" * w + "┘")
```
Box-drawing characters create the ASCII box:
- Top: `┌──────────┐`
- Content: `│ content  │`
- Bottom: `└──────────┘`

**Arrows:**
```python
        cons = consumers.get(nid, [])
        if cons:
            if len(cons) == 1:
                lines.append("       │")
                lines.append("       ▼")
            else:
                c_ids = ", ".join(node_sid[id(c)] for c in cons)
                lines.append(f"       │ ──► feeds {c_ids}")
                lines.append("       ▼")
```
If a node feeds into multiple consumers (DAG branch), list their short IDs.

---

### `_render_dot(nodes, edges, show_grads)` — Graphviz DOT Renderer

```python
dot_id: Dict[int, str] = {id(n): f"n{i}" for i, n in enumerate(nodes)}
```
Graphviz node names must start with a letter, hence the `n` prefix.

```python
lines: List[str] = [
    "digraph neural_network_frameworkComputationGraph {",
    '  graph [rankdir=TB, splines=ortho, bgcolor="#F8F9FA", ...];',
    '  node  [style="filled,rounded", ...];',
    '  edge  [color="#555555", arrowsize=0.75, ...];',
]
```
DOT file structure:
- `digraph` = directed graph
- `rankdir=TB` = top-to-bottom layout (inputs at top, loss at bottom)
- `splines=ortho` = orthogonal (right-angle) edge routing, cleaner for many nodes
- `style="filled,rounded"` = rounded rectangles with fill colours

**Node declarations:**
```python
    lines.append(
        f'  {nid} [label="{label}", shape={shape}, '
        f'fillcolor="{color}", penwidth={penwidth}, color="#333333"];'
    )
```
Each node becomes a DOT node with op name, shape, optional gradient norm, fill colour,
and border thickness (thicker for learnable parameters).

**Edge declarations:**
```python
    lines.append(f'  {src_dot} -> {dst_dot} [label="{edge_label}"];')
```
`->` is the DOT directed-edge syntax. Each edge is labelled with the tensor's shape.

---

### `draw_graph(tensor, fmt, show_grads, path, render_svg)` — Public API

The single public function. Everything else in `viz.py` has an underscore prefix (private).

```python
    if not isinstance(tensor, Tensor):
        raise TypeError(
            f"draw_graph() expects a neural_network_framework Tensor, got {type(tensor).__name__!r}"
        )
```
Type-check the input. If someone passes a NumPy array by mistake, they get a clear error.
`{type(tensor).__name__!r}` formats the type name as a repr string: `'ndarray'`.

```python
    nodes, edges = _collect(tensor)

    if fmt == "dot":
        output = _render_dot(nodes, edges, show_grads)
        ext = ".dot"
    elif fmt == "ascii":
        output = _render_ascii(tensor, nodes, edges, show_grads)
        ext = ".txt"
    else:
        raise ValueError(f"fmt must be 'ascii' or 'dot', got {fmt!r}")
```
Collect the graph, dispatch to the appropriate renderer, or raise a clear error for
an unrecognized format.

```python
    if path is not None:
        with open(path + ext, "w", encoding="utf-8") as fh:
            fh.write(output)

        if render_svg and fmt == "dot":
            ret = os.system(f'dot -Tsvg "{path}.dot" -o "{path}.svg"')
            if ret == 0:
                print(f"[neural_network_framework.viz] SVG rendered → '{path}.svg'")
            else:
                print("[neural_network_framework.viz] 'dot' not found — install Graphviz ...")
    else:
        print(output)
```

`with open(...) as fh:` — the `with` statement ensures the file is always closed,
even if an exception occurs mid-write.

`os.system(...)` runs a shell command. Returns 0 on success, nonzero on failure.
Graceful degradation: if Graphviz is not installed, the DOT file is still saved
and a helpful message is printed.

**Using `draw_graph` — four common scenarios:**

```python
# 1. See the graph structure before backward (no gradient info)
draw_graph(loss)

# 2. After backward: see gradient norms and health flags
loss.backward()
draw_graph(loss, show_grads=True)

# 3. Save to file (useful for large graphs)
draw_graph(loss, path="my_graph")          # → my_graph.txt

# 4. Export to Graphviz (for SVG/PNG rendering)
draw_graph(loss, fmt="dot", path="my_graph", render_svg=True)
# → saves my_graph.dot + my_graph.svg (if graphviz is installed)
# → to render manually: dot -Tsvg my_graph.dot -o my_graph.svg
```

---

## 14. A Complete Training Step — Everything Together

Here is exactly what happens, step by step, when you execute one training iteration.

```python
logits = model(X_batch)
```

1. `model.__call__(X_batch)` → `model.forward(X_batch)` (via `Module.__call__`)
2. If `model` is `Sequential`, it loops through `self.layers` in order
3. Each layer's `forward` computes its output and attaches a `_backward` closure
4. Every `@`, `+`, activation creates a new Tensor connected via `._prev`
5. A computation graph grows in memory

At the end: `logits` is a Tensor with shape `(batch, num_classes)` and a full graph behind it.

```python
loss = criterion(logits, y_batch)
```

1. Loss function computes a single scalar via log-softmax (or whatever loss is used)
2. `loss.requires_grad = True` (inherited from `logits`)
3. `loss.data` is a scalar — "how wrong is the model right now?"

```python
model.zero_grad()
```

1. Calls `p.zero_grad()` for every parameter
2. `p.grad.fill(0.0)` — in-place zero, no new memory
3. MUST happen before backward, not after

```python
loss.backward()
```

1. `loss._init_grad()` → `loss.grad = array([1.0])` (∂loss/∂loss = 1)
2. `build_topo(loss)` — DFS to collect all nodes in topological order
3. `for node in reversed(topo): node._backward()`
4. Each closure reads `node.grad` (fully accumulated) and `+=` into parents' `.grad`
5. After the full loop: `layer.weight.grad` = `∂loss/∂weight` for every layer

```python
optimizer.step()
```

1. For each parameter: read `.grad`, apply update rule (Adam, SGD, etc.)
2. `p.data -= lr × effective_gradient` — the actual weight update
3. The model has improved slightly on this batch

**The computation graph is discarded automatically** when Python garbage-collects
the intermediate tensor objects. A brand new graph is built on the next forward pass.
This is a **dynamic computation graph** — rebuilt from scratch every step, which is why
`if` statements and loops inside `forward()` work correctly with autograd.

---

## 15. Python Mechanics Master Reference

### Closures — Captured Variables

```python
def make_multiplier(n):
    def multiply(x):
        return x * n    # n is captured from make_multiplier's scope
    return multiply

times3 = make_multiplier(3)
times3(10)   # → 30 — even though make_multiplier has returned
```

**Warning:** Python closures capture the variable NAME, not the value.
```python
funcs = []
for i in range(3):
    funcs.append(lambda: i)

[f() for f in funcs]   # [2, 2, 2] — all see the final value of i!

# Fix: capture the value explicitly
funcs = []
for i in range(3):
    funcs.append(lambda x=i: x)

[f() for f in funcs]   # [0, 1, 2] — correct
```

This is why BatchNorm's backward closures use `_mu = mu`, `_inv_std = inv_std`, etc.

---

### Generators

```python
def batch_generator(data, batch_size):
    start = 0
    while start < len(data):
        yield data[start:start+batch_size]
        start += batch_size

for batch in batch_generator(range(10), 3):
    print(list(batch))
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]
```

`yield` pauses the function and sends a value to the caller.
The function's state (local variables, position in the loop) is preserved.
On the next `next()` call (or loop iteration), it resumes exactly where it paused.

`yield from gen` is shorthand for `for x in gen: yield x`.
Used in `named_parameters` to delegate yielding to a sub-generator.

---

### `isinstance` vs `type`

```python
class Animal: pass
class Dog(Animal): pass

dog = Dog()
isinstance(dog, Animal)   # True  — Dog IS-A Animal (subclass)
type(dog) == Animal       # False — exact type is Dog, not Animal
```

Always prefer `isinstance` in production code — it respects inheritance.

---

### f-string Format Specifiers

```python
value = 1234567.891
i = 7
name = "hello"

f"{value:.4f}"     # "1234567.8910" — 4 decimal places
f"{value:,.2f}"    # "1,234,567.89" — comma separator + 2 decimal places
f"#{i:02d}"        # "#07"          — zero-padded to 2 digits
f"{name:<10}"      # "hello     "   — left-align, pad to 10 chars
f"{name:>10}"      # "     hello"   — right-align, pad to 10 chars
f"{name:^10}"      # "  hello   "   — center-align, pad to 10 chars
f"{type(x).__name__!r}"   # "'int'"  — repr of the type name
```

---

### Context Managers (`with`)

```python
with open("file.txt", "w", encoding="utf-8") as f:
    f.write("hello")
# File is guaranteed to be closed here, even if an exception occurred
```

The `with` statement calls `f.__enter__()` before the block and `f.__exit__()` after.
For file objects, `__exit__` closes the file. Always use `with` when working with files.

---

### NumPy Broadcasting

When you add arrays of different shapes, NumPy broadcasts the smaller one:

```python
a = np.array([[1,2,3],[4,5,6]])   # shape (2,3)
b = np.array([10,20,30])           # shape (3,)
(a + b).shape                      # (2,3) — b is broadcast across rows
```

Rules:
1. Align shapes from the right
2. Dimensions of size 1 are stretched to match the other
3. Dimensions that are missing (fewer dims) get a leading size-1

`_unbroadcast` reverses this in the backward pass by summing over the broadcast axes.

---

### `np.add.at` — Unbuffered Scatter Addition

```python
arr = np.zeros(5)
idx = [1, 1, 2]

# WRONG — numpy buffers the indexed += update:
arr[idx] += np.array([10, 20, 30])
# arr[1] = 20, NOT 30 (only the second += applied)

# CORRECT — unbuffered, accumulates properly:
np.add.at(arr, idx, np.array([10, 20, 30]))
# arr[1] = 30 (10+20 correctly accumulated)
```

Critical for `Embedding._backward` where the same embedding row can receive
gradients from multiple token positions in one batch.

---

### `np.broadcast_to` — Zero-Copy View

```python
a = np.array([1, 2, 3])              # shape (3,)
b = np.broadcast_to(a, (4, 3))       # shape (4,3) — NO data copied
b[0]   # [1, 2, 3]
b[1]   # [1, 2, 3]
```

`broadcast_to` creates a view with stride tricks — the same data is "seen" at multiple
positions without copying. Used in backward passes to spread a gradient to a larger shape
efficiently.

---

## 16. Math Reference

### Derivative

The derivative `df/dx` is the rate of change of `f` with respect to `x`:

`df/dx = lim_{h→0} (f(x+h) - f(x)) / h`

Common derivatives:
```
f(x) = xⁿ     →  n × x^(n-1)         power rule
f(x) = eˣ     →  eˣ                   exp is its own derivative
f(x) = log(x) →  1/x
f(x) = sin(x) →  cos(x)
f(x) = c      →  0                    constants have zero derivative
```

### Chain Rule

For composed functions `f(g(x))`:
```
df/dx = (df/dg) × (dg/dx)
```

Multiple composition: `L(f(g(h(x))))`:
```
∂L/∂x = (∂L/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

This is backpropagation. Each `×` is one backward closure call.

### Matrix Multiplication Gradient

For `C = A @ B`:
```
∂loss/∂A = ∂loss/∂C @ B^T     (multiply by B transposed on the right)
∂loss/∂B = A^T @ ∂loss/∂C     (multiply by A transposed on the left)
```

Dimension check for A=(m,n), B=(n,p), ∂loss/∂C=(m,p):
- `∂loss/∂A`: `(m,p) @ (p,n) = (m,n)` ✓ matches A's shape
- `∂loss/∂B`: `(n,m) @ (m,p) = (n,p)` ✓ matches B's shape

### Softmax Gradient

For `s = softmax(x)`:
```
∂loss/∂x = s ⊙ (∂loss/∂s - dot(∂loss/∂s, s))
```
Where `⊙` is element-wise multiplication and `dot` is the scalar dot product per row.
The subtraction term accounts for the constraint that softmax outputs must sum to 1.

### Cross-Entropy Gradient

For `L = CrossEntropy(logits, target)`:
```
∂L/∂logitsᵢ = softmax(logits)ᵢ - 1{i == target}
```
Subtract 1 from the correct class's softmax score. All other classes are unchanged.
The cleanest gradient in deep learning.

### Xavier Initialization

Goal: preserve variance across layers.
For linear layer with ReLU-free activation:
```
Var(W) = 2 / (fan_in + fan_out)
```
For Uniform(-a, a): Var = a²/3, so `a = sqrt(6 / (fan_in + fan_out))`.

### He Initialization

Goal: preserve variance through ReLU layers (which zero ~50% of neurons).
```
Var(W) = 2 / fan_in
```
Doubles Xavier's variance to compensate for ReLU's variance halving.

### Adam Bias Correction

Moments start at zero. After 1 step: `m₁ = (1-β₁) × g` — much smaller than true mean.
Bias correction: `m̂₁ = m₁ / (1 - β₁¹) = (1-β₁)g / (1-β₁) = g`. Correct!

At step t: `m̂ₜ = mₜ / (1 - β₁ᵗ)`. As t→∞, `β₁ᵗ→0`, correction disappears.
Makes Adam correct from the very first step, regardless of how `β₁` is set.

---

*Every line in neural_network_framework has a reason. This document has explained all of them.*
*If anything is still unclear, re-read the relevant section — it is all here.*
