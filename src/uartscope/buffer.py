"""Fixed-size ring buffer for multi-channel samples."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Keeps the last ``capacity`` samples of ``n_channels`` channels."""

    def __init__(self, capacity: int, n_channels: int = 1):
        self.capacity = int(capacity)
        self.n_channels = int(n_channels)
        self._data = np.zeros((self.capacity, self.n_channels))
        self._pos = 0  # next write index
        self.total = 0  # samples written since creation

    def append(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=float).reshape(-1, self.n_channels)
        if len(chunk) >= self.capacity:
            chunk = chunk[-self.capacity:]
        n = len(chunk)
        end = self._pos + n
        if end <= self.capacity:
            self._data[self._pos:end] = chunk
        else:
            first = self.capacity - self._pos
            self._data[self._pos:] = chunk[:first]
            self._data[: n - first] = chunk[first:]
        self._pos = end % self.capacity
        self.total += n

    def __len__(self) -> int:
        return min(self.total, self.capacity)

    def get(self) -> np.ndarray:
        """Return the stored samples in time order, shape (len, n_channels)."""
        if self.total < self.capacity:
            return self._data[: self.total].copy()
        return np.roll(self._data, -self._pos, axis=0)
