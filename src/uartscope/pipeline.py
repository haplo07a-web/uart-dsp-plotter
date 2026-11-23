"""Source -> filter -> buffers (-> CSV). Shared by the live plot and the snapshot mode."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .buffer import RingBuffer
from .dsp import StreamingFilter
from .export import CsvRecorder


class Pipeline:
    def __init__(
        self,
        source,
        fs: float,
        window_s: float,
        filt_kwargs: dict,
        csv_path: Optional[str] = None,
        n_channels: Optional[int] = None,
        min_samples: int = 0,
    ):
        self.source = source
        self.fs = float(fs)
        self.window_n = max(int(window_s * self.fs), 16)  # samples shown in the time plot
        self.capacity = max(self.window_n, int(min_samples))  # kept in memory (>= FFT size)
        self.filt_kwargs = dict(filt_kwargs)
        self.csv_path = csv_path
        self.n_channels = n_channels
        self.raw: Optional[RingBuffer] = None
        self.filt: Optional[RingBuffer] = None
        self.filter: Optional[StreamingFilter] = None
        self.recorder: Optional[CsvRecorder] = None
        self._rate_t = time.monotonic()
        self._rate_n = 0
        self.measured_rate = float("nan")  # samples/s actually arriving
        if n_channels is not None:
            self._setup(n_channels)

    def _setup(self, n_channels: int) -> None:
        """Create buffers/filter once the channel count is known (serial: after the first line)."""
        self.n_channels = n_channels
        self.raw = RingBuffer(self.capacity, n_channels)
        self.filt = RingBuffer(self.capacity, n_channels)
        self.filter = StreamingFilter(fs=self.fs, n_channels=n_channels, **self.filt_kwargs)
        if self.csv_path:
            self.recorder = CsvRecorder(self.csv_path, self.fs, n_channels)

    def step(self) -> int:
        """Read whatever the source has, process it, return number of new samples."""
        chunk = self.source.read()
        if len(chunk) == 0:
            return 0
        if self.raw is None:
            self._setup(chunk.shape[1])
        y = self.filter.process(chunk)
        self.raw.append(chunk)
        self.filt.append(y)
        if self.recorder is not None:
            self.recorder.write(chunk, y)
        self._update_rate(len(chunk))
        return len(chunk)

    def _update_rate(self, n: int) -> None:
        self._rate_n += n
        now = time.monotonic()
        dt = now - self._rate_t
        if dt >= 1.0:
            self.measured_rate = self._rate_n / dt
            self._rate_t, self._rate_n = now, 0

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        self.source.close()
