# neural_network_framework
<div align="center">

# 🧠 neural_network_framework

**A neural network framework built from scratch — in pure Python + NumPy.**

*Every line of gradient math is written explicitly. No black boxes.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/Requires-NumPy-orange?logo=numpy)](https://numpy.org)
[![Tests](https://img.shields.io/badge/Tests-57%20passed-brightgreen?logo=pytest)](tests/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

</div>

---

## What is neural_network_framework?

`neural_network_framework` is a minimal, educational deep learning framework that implements everything from first principles:

- **Reverse-mode automatic differentiation** (backprop) — a real compute graph, topological sort, and gradient closures
- **Trainable layers** — `Linear`, `BatchNorm1d`, `Dropout`, `Embedding`
- **Activations** — `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`
- **Loss functions** — `MSE`, `BCE`, `CrossEntropy` (numerically stable), `NLL`
- **Optimizers** — `SGD` (+ momentum/Nesterov), `Adam`, `AdamW`
- **Data utilities** — `Dataset`, `TensorDataset`, `DataLoader`

The codebase is intentionally small (~1 200 lines), heavily commented, and mirrors the PyTorch API so it's easy to follow.

---

## Project Structure

```
neural_network_framework/
│
├── neural_network_framework/                  ← framework package
│   ├── tensor.py              ← Tensor class + autograd engine ⭐
│   ├── loss.py                ← MSE, BCE, CrossEntropy, NLL
│   │
│   ├── nn/
│   │   ├── module.py          ← Module, Sequential base classes
│   │   ├── layers.py          ← Linear, BatchNorm1d, Dropout, Embedding
│   │   └── activations.py     ← ReLU, Sigmoid, Tanh, GELU, Softmax …
│   │
│   ├── optim/
│   │   ├── sgd.py             ← SGD + momentum + Nesterov + weight decay
│   │   └── adam.py            ← Adam (AMSGrad) + AdamW
│   │
│   └── utils/
│       ├── data.py            ← Dataset / DataLoader
│       └── init.py            ← Xavier / He / Orthogonal initialisers
│
├── examples/
│   ├── 01_xor.py              ← XOR gate — why depth matters
│   ├── 02_mnist.py            ← MNIST handwritten digits (~97% acc)
│   └── 03_regression.py       ← Sine-wave regression + matplotlib plot
│
├── tests/
│   ├── test_tensor.py         ← Gradient checks via finite differences
│   ├── test_layers.py         ← Layer / activation / loss correctness
│   └── test_optim.py          ← SGD, Adam, AdamW convergence tests
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install numpy pytest

# Optional (only needed for specific examples)
pip install matplotlib     # for 03_regression.py
pip install scikit-learn   # for 02_mnist.py
```

No build step, no compilation, no CUDA. Works on any OS.

### 2. Run the examples

```bash
# XOR gate — trains to 100% accuracy in ~200 steps
python examples/01_xor.py

# Sine-wave regression (saves a plot to examples/sine_regression.png)
python examples/03_regression.py

# MNIST classifier — ~97% test accuracy in 10 epochs
python examples/02_mnist.py
```

### 3. Run the tests

```bash
python -m pytest tests/ -v
```

```
57 passed in 0.23s ✅
```

---

## Quick-Start Code

```python
import numpy as np
import neural_network_framework.nn as nn
from neural_network_framework.tensor import Tensor
from neural_network_framework.optim import Adam
from neural_network_framework.loss import CrossEntropyLoss
from neural_network_framework.utils.data import TensorDataset, DataLoader

# ── 1. Build a model ──────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)

# ── 2. Choose optimizer & loss ────────────────────────────────────────
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = CrossEntropyLoss()

# ── 3. Create a DataLoader ────────────────────────────────────────────
dataset = TensorDataset(X_train, y_train)          # numpy arrays
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

# ── 4. Training loop ──────────────────────────────────────────────────
model.train()
for X_batch, y_batch in loader:
    logits = model(X_batch)           # forward pass
    loss   = criterion(logits, y_batch)

    model.zero_grad()                 # clear old gradients
    loss.backward()                   # backprop through the graph
    optimizer.step()                  # update weights
```

---

## How Autograd Works

Every tensor operation returns a new `Tensor` and attaches a `_backward` closure:

```
x ──[Linear]──► h ──[ReLU]──► y ──[CrossEntropy]──► loss
                                                        │
                                              loss.backward()
                                                        │
                              ◄── ∂loss/∂y ◄── ∂loss/∂h ◄── ∂loss/∂x
```

`loss.backward()` performs a **topological sort** of the computation graph,
then calls each `_backward` closure in reverse order — the chain rule, made explicit.

```python
# A peek inside tensor.py
def __add__(self, other):
    out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other))

    def _backward():
        self.grad  += _unbroadcast(out.grad, self.shape)   # ∂out/∂self  = 1
        other.grad += _unbroadcast(out.grad, other.shape)  # ∂out/∂other = 1

    out._backward = _backward
    return out
```

---

## API Reference

### Tensor

```python
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Arithmetic
x + y  |  x - y  |  x * y  |  x / y  |  x ** 2  |  x @ y  |  -x

# Math
x.exp()  |  x.log()  |  x.sqrt()  |  x.abs()

# Reductions
x.sum(axis=None)  |  x.mean(axis=None)  |  x.max(axis=None)

# Shape
x.reshape(4)  |  x.T  |  x.flatten()

# Backward
loss.backward()     # triggers full graph backprop
print(x.grad)       # ∂loss/∂x
x.zero_grad()       # reset gradient
```

### Layers

| Layer | Signature | Notes |
|---|---|---|
| `Linear` | `(in, out, bias=True, init='he')` | He init by default (good for ReLU) |
| `BatchNorm1d` | `(features, eps=1e-5, momentum=0.1)` | tracks running stats for eval |
| `Dropout` | `(p=0.5)` | inverted dropout, disabled in eval mode |
| `Embedding` | `(vocab_size, embed_dim)` | sparse gradient accumulation |

### Activations

```python
nn.ReLU()  |  nn.LeakyReLU(0.01)  |  nn.Sigmoid()  |  nn.Tanh()
nn.GELU()  |  nn.Softmax(axis=-1)  |  nn.LogSoftmax(axis=-1)
```

### Loss Functions

| Loss | Use case |
|---|---|
| `MSELoss()` | Regression |
| `BCEWithLogitsLoss()` | Binary classification (stable sigmoid + BCE) |
| `CrossEntropyLoss()` | Multi-class classification (stable log-softmax + NLL) |
| `NLLLoss()` | Paired with `LogSoftmax` |

### Optimizers

```python
# SGD with momentum + Nesterov
SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

# Adam
Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW — decoupled weight decay (recommended over Adam for regularisation)
AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

---

## Example Results

### XOR Gate (`examples/01_xor.py`)

```
  Step        Loss    Accuracy
     1    0.704681       50.0%
   200    0.001281      100.0%
  2000    0.000030      100.0%

0 XOR 0 = 0  →  p=0.0000  [✓]
0 XOR 1 = 1  →  p=1.0000  [✓]
1 XOR 0 = 1  →  p=1.0000  [✓]
1 XOR 1 = 0  →  p=0.0001  [✓]
```

### MNIST (`examples/02_mnist.py`) — 10 epochs, Adam, batch 256

```
Epoch    Train Loss    Train Acc    Test Acc
    1        0.3021       90.81%      95.12%
    5        0.0891       97.24%      96.98%
   10        0.0512       98.41%      97.63%
```

---

## Design Principles

| Principle | How it's achieved |
|---|---|
| **Readability** | Every gradient formula has its math written out inline |
| **No hidden state** | All backward values captured explicitly in closures |
| **PyTorch-compatible API** | Same method names — easy to migrate |
| **Correctness** | 57 tests, gradient checks via central-difference finite differences |
| **Zero magic** | No metaclasses, no C extensions, no hidden registries |

---

## Extending neural_network_framework

Adding a new op is just three steps:

```python
# 1. Compute the forward value
val = np.your_op(x.data)

# 2. Create the output Tensor
out = Tensor(val, requires_grad=x.requires_grad, _children=(x,), _op="YourOp")

# 3. Attach the backward closure (∂out/∂x goes in x.grad)
def _backward():
    if x.requires_grad:
        x._init_grad()
        x.grad += out.grad * your_local_derivative

out._backward = _backward
return out
```

Adding a new layer means subclassing `Module` and implementing `forward()`. The `parameters()` method is auto-discovered via Python's `vars()` — no registration needed.

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | All tensor math |
| `pytest` | ≥ 7.0 | Running tests |
| `matplotlib` | ≥ 3.5 | *(optional)* regression plot |
| `scikit-learn` | ≥ 1.0 | *(optional)* MNIST download |

---

## License

MIT — free to use, modify, and learn from.
