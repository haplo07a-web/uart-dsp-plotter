"""Sample sources: a real serial port, or a simulated signal generator."""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from .parser import LineAssembler, parse_line


class SimSource:
    """Signal generator: sum of sines plus white Gaussian noise.

    realtime=True:  each read() returns the samples "produced" since the last read
                    according to the wall clock, like a real device streaming at fs.
    realtime=False: each read() returns exactly ``chunk`` samples (for tests/snapshots).
    Channel i gets the same tones with a phase shift of i * 90 degrees.
    """

    def __init__(
        self,
        fs: float = 1000.0,
        freqs: Sequence[float] = (10.0, 50.0),
        amps: Sequence[float] = (1.0, 0.5),
        noise: float = 0.2,
        n_channels: int = 1,
        realtime: bool = True,
        chunk: int = 50,
        seed: Optional[int] = None,
    ):
        if len(freqs) != len(amps):
            raise ValueError("freqs and amps must have the same length")
        self.fs = float(fs)
        self.freqs = np.asarray(freqs, dtype=float)
        self.amps = np.asarray(amps, dtype=float)
        self.noise = float(noise)
        self.n_channels = int(n_channels)
        self.realtime = realtime
        self.chunk = int(chunk)
        self._rng = np.random.default_rng(seed)
        self._n = 0  # samples produced so far
        self._t0 = time.monotonic()

    def generate(self, n: int) -> np.ndarray:
        """Produce the next ``n`` samples, shape (n, n_channels)."""
        t = (self._n + np.arange(n)) / self.fs
        out = np.empty((n, self.n_channels))
        for ch in range(self.n_channels):
            phase = ch * np.pi / 2
            tones = self.amps[:, None] * np.sin(2 * np.pi * self.freqs[:, None] * t + phase)
            out[:, ch] = tones.sum(axis=0) + self.noise * self._rng.standard_normal(n)
        self._n += n
        return out

    def read(self) -> np.ndarray:
        if self.realtime:
            due = int((time.monotonic() - self._t0) * self.fs) - self._n
            n = max(due, 0)
        else:
            n = self.chunk
        return self.generate(n)

    def close(self) -> None:
        pass


class SerialSource:
    """Read text lines from a serial port without blocking.

    The first valid line fixes the channel count; later lines with a different
    number of values are ignored (they are usually corrupted).
    """

    def __init__(self, port: str, baud: int = 115200, n_channels: Optional[int] = None):
        import serial  # imported here so the simulator works without pyserial

        self._ser = serial.Serial(port, baud, timeout=0)
        self._lines = LineAssembler()
        self.n_channels = n_channels
        self.dropped = 0  # lines that could not be parsed

    def read(self) -> np.ndarray:
        waiting = self._ser.in_waiting
        data = self._ser.read(waiting) if waiting else b""
        rows = []
        for line in self._lines.feed(data):
            values = parse_line(line, self.n_channels)
            if values is None:
                if line.strip():
                    self.dropped += 1
                continue
            if self.n_channels is None:
                self.n_channels = len(values)
            rows.append(values)
        n_ch = self.n_channels or 1
        return np.asarray(rows, dtype=float).reshape(-1, n_ch)

    def close(self) -> None:
        self._ser.close()


def list_ports() -> list:
    """Return (device, description) for each serial port on this computer."""
    from serial.tools import list_ports as lp

    return [(p.device, p.description) for p in lp.comports()]
