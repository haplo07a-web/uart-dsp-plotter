"""CSV recording of raw and filtered samples."""

from __future__ import annotations

import csv
from typing import Optional, TextIO

import numpy as np


class CsvRecorder:
    """Write samples to CSV: t_s, ch0_raw, ch0_filt, ch1_raw, ch1_filt, ...

    Time is sample_index / fs (the device is assumed to sample at a fixed rate).
    """

    def __init__(self, path: str, fs: float, n_channels: int):
        self.path = path
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self._file: Optional[TextIO] = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        header = ["t_s"]
        for ch in range(self.n_channels):
            header += [f"ch{ch}_raw", f"ch{ch}_filt"]
        self._writer.writerow(header)
        self.rows = 0

    def write(self, raw: np.ndarray, filt: np.ndarray) -> None:
        raw = np.asarray(raw).reshape(-1, self.n_channels)
        filt = np.asarray(filt).reshape(-1, self.n_channels)
        t = (self.rows + np.arange(len(raw))) / self.fs
        inter = np.empty((len(raw), 2 * self.n_channels))
        inter[:, 0::2] = raw
        inter[:, 1::2] = filt
        for ti, row in zip(t, inter):
            self._writer.writerow([f"{ti:.6f}"] + [f"{v:.6g}" for v in row])
        self.rows += len(raw)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "CsvRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
