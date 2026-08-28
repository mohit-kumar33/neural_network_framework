"""
data.py — Lightweight Dataset and DataLoader utilities.

Designed to be dependency-free (only NumPy) while providing the same
ergonomic API as PyTorch's DataLoader.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Iterator, Optional

from neural_network_framework.tensor import Tensor


class Dataset:
    """
    Abstract base class for datasets.

    Subclass this and implement `__len__` and `__getitem__`.

    Example
    -------
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    A Dataset wrapping a pair (or tuple) of NumPy arrays / Tensors.

    All arrays must have the same first dimension (number of samples).
    """

    def __init__(self, *arrays):
        self.arrays = []
        for a in arrays:
            if isinstance(a, Tensor):
                self.arrays.append(a.data)
            else:
                self.arrays.append(np.array(a))

        sizes = [a.shape[0] for a in self.arrays]
        assert len(set(sizes)) == 1, "All arrays must have the same first dimension."

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, idx):
        return tuple(a[idx] for a in self.arrays)


class DataLoader:
    """
    Iterates over a Dataset in mini-batches.

    Parameters
    ----------
    dataset : Dataset
        The dataset to iterate over.
    batch_size : int
        Number of samples per batch. Default: 32.
    shuffle : bool
        If True, data is shuffled at the start of every epoch. Default: True.
    drop_last : bool
        If True, the last incomplete batch is dropped. Default: False.

    Usage
    -----
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for X_batch, y_batch in loader:
        # X_batch and y_batch are Tensors
        ...
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        start = 0
        while start < n:
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start:end]

            # Collect items and convert to Tensors
            items = [self.dataset[i] for i in batch_idx]
            if isinstance(items[0], (tuple, list)):
                # Multiple arrays per sample
                batch = tuple(
                    Tensor(np.stack([item[j] for item in items]))
                    for j in range(len(items[0]))
                )
            else:
                batch = Tensor(np.stack(items))

            yield batch
            start = end
