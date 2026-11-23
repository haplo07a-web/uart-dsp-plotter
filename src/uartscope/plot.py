"""Live matplotlib view: time domain (raw + filtered) on top, FFT spectrum below.

pyplot is imported inside functions so the caller can choose the backend first
(e.g. "Agg" for headless snapshots).
"""

from __future__ import annotations

import numpy as np

from .dsp import peak_frequency, spectrum, to_db
from .pipeline import Pipeline


class Scope:
    def __init__(self, pipeline: Pipeline, fft_size: int = 2048, fft_channel: int = 0, title: str = ""):
        import matplotlib.pyplot as plt

        self.p = pipeline
        self.fft_size = int(fft_size)
        self.fft_channel = int(fft_channel)
        self.title = title
        self.paused = False
        self.fig, (self.ax_t, self.ax_f) = plt.subplots(2, 1, figsize=(10, 7))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("uartscope")
        self.ax_t.set_xlabel("time (s)")
        self.ax_t.set_ylabel("value")
        self.ax_t.grid(True, alpha=0.3)
        self.ax_f.set_xlabel("frequency (Hz)")
        self.ax_f.set_ylabel("amplitude (dB)")
        self.ax_f.grid(True, alpha=0.3)
        self.ax_f.set_xlim(0, self.p.fs / 2)
        self._lines_raw, self._lines_filt = [], []
        self._spec_raw = self._spec_filt = None
        self._peak_text = self.ax_f.text(
            0.99, 0.95, "", transform=self.ax_f.transAxes, ha="right", va="top", family="monospace"
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.tight_layout()

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.paused = not self.paused

    def _create_lines(self, n_channels: int) -> None:
        for ch in range(n_channels):
            color = f"C{ch % 10}"
            (lr,) = self.ax_t.plot([], [], color=color, alpha=0.35, lw=0.8, label=f"ch{ch} raw")
            (lf,) = self.ax_t.plot([], [], color=color, lw=1.5, label=f"ch{ch} filtered")
            self._lines_raw.append(lr)
            self._lines_filt.append(lf)
        (self._spec_raw,) = self.ax_f.plot([], [], color="0.6", lw=0.8, label="raw")
        (self._spec_filt,) = self.ax_f.plot([], [], color="C3", lw=1.2, label="filtered")
        self.ax_t.legend(loc="upper right", fontsize=8, ncol=min(n_channels, 4))
        self.ax_f.legend(loc="lower left", fontsize=8)

    def refresh(self) -> None:
        """Redraw lines from the current buffer contents."""
        if self.p.raw is None or len(self.p.raw) < 2:
            return
        if not self._lines_raw:
            self._create_lines(self.p.n_channels)
        raw, filt = self.p.raw.get(), self.p.filt.get()
        n = len(raw)
        w = min(n, self.p.window_n)
        t = (self.p.raw.total - w + np.arange(w)) / self.p.fs
        for ch in range(self.p.n_channels):
            self._lines_raw[ch].set_data(t, raw[-w:, ch])
            self._lines_filt[ch].set_data(t, filt[-w:, ch])
        self.ax_t.set_xlim(t[0], max(t[0] + self.p.window_n / self.p.fs, t[-1]))
        self.ax_t.relim()
        self.ax_t.autoscale_view(scalex=False)

        ch = min(self.fft_channel, self.p.n_channels - 1)
        m = min(self.fft_size, n)
        if m >= 16:
            f, a_raw = spectrum(raw[-m:, ch], self.p.fs)
            _, a_filt = spectrum(filt[-m:, ch], self.p.fs)
            d_raw, d_filt = to_db(a_raw), to_db(a_filt)
            self._spec_raw.set_data(f, d_raw)
            self._spec_filt.set_data(f, d_filt)
            top = max(d_raw.max(), d_filt.max())
            self.ax_f.set_ylim(top - 100, top + 10)
            fpk = peak_frequency(f, a_filt, fmin=f[1])
            self._peak_text.set_text(f"peak ch{ch}: {fpk:8.2f} Hz\nFFT: {m} pts, df={self.p.fs / m:.2f} Hz")

        rate = self.p.measured_rate
        rate_txt = f", arriving {rate:.0f} S/s" if np.isfinite(rate) else ""
        paused = "  [PAUSED]" if self.paused else ""
        self.ax_t.set_title(f"{self.title}  |  fs={self.p.fs:g} Hz{rate_txt}{paused}", fontsize=10)

    def _update(self, _frame):
        self.p.step()  # always drain the source, even when paused
        if not self.paused:
            self.refresh()
        return []

    def run(self, interval_ms: int = 50) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self._anim = FuncAnimation(self.fig, self._update, interval=interval_ms, cache_frame_data=False)
        plt.show()

    def save(self, path: str) -> None:
        self.refresh()
        self.fig.savefig(path, dpi=110)
