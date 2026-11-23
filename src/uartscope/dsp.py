"""Streaming digital filters and spectrum estimation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal

Cutoff = Union[float, Sequence[float]]

FILTER_KINDS = ("none", "lowpass", "highpass", "bandpass", "bandstop", "notch")


class StreamingFilter:
    """Filter a signal chunk by chunk, keeping the filter state between chunks.

    Filtering many small chunks gives the same result as filtering the whole
    signal at once, so the live plot shows no glitches at chunk borders.

    kind:   one of FILTER_KINDS
    cutoff: Hz; a single value for lowpass/highpass/notch, (low, high) for band filters
    method: "butter" (IIR Butterworth, second-order sections) or "fir" (windowed FIR)
    """

    def __init__(
        self,
        kind: str,
        fs: float,
        cutoff: Optional[Cutoff] = None,
        order: int = 4,
        method: str = "butter",
        numtaps: int = 101,
        q: float = 30.0,
        n_channels: int = 1,
        steady_start: bool = True,
    ):
        if kind not in FILTER_KINDS:
            raise ValueError(f"unknown filter kind {kind!r}; choose from {FILTER_KINDS}")
        self.kind = kind
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.steady_start = steady_start
        self._sos = None
        self._fir = None
        if kind == "none":
            return
        if cutoff is None:
            raise ValueError(f"filter kind {kind!r} needs a cutoff frequency")
        if kind in ("bandpass", "bandstop"):
            cutoff = tuple(float(c) for c in cutoff)  # type: ignore[union-attr]
            if len(cutoff) != 2 or not 0 < cutoff[0] < cutoff[1]:
                raise ValueError("band filters need cutoff=(low, high) with 0 < low < high")
        else:
            cutoff = float(cutoff)  # type: ignore[arg-type]
        nyq = self.fs / 2
        if max(np.atleast_1d(cutoff)) >= nyq:
            raise ValueError(f"cutoff must be below fs/2 = {nyq:g} Hz")

        if kind == "notch":
            b, a = signal.iirnotch(cutoff, q, fs=self.fs)
            self._sos = signal.tf2sos(b, a)
        elif method == "butter":
            self._sos = signal.butter(order, cutoff, btype=kind, fs=self.fs, output="sos")
        elif method == "fir":
            if numtaps % 2 == 0:
                numtaps += 1  # odd length: type I FIR, works for every band type
            pass_zero = kind in ("lowpass", "bandstop")
            self._fir = signal.firwin(numtaps, cutoff, pass_zero=pass_zero, fs=self.fs)
        else:
            raise ValueError(f"unknown method {method!r}; choose 'butter' or 'fir'")
        self.reset()

    def reset(self) -> None:
        """Forget the filter state (next chunk starts fresh)."""
        self._zi = None

    def _initial_state(self, x0: np.ndarray) -> np.ndarray:
        """Filter state for the first chunk; scaled by the first sample if steady_start."""
        scale = x0 if self.steady_start else np.zeros_like(x0)
        if self._sos is not None:
            zi = signal.sosfilt_zi(self._sos)  # (n_sections, 2)
            return zi[:, :, None] * scale[None, None, :]  # (n_sections, 2, n_ch) for axis=0
        zi = signal.lfilter_zi(self._fir, 1.0)  # (numtaps-1,)
        return zi[:, None] * scale[None, :]  # (numtaps-1, n_ch)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter ``chunk`` of shape (n_samples, n_channels) and return the same shape."""
        x = np.asarray(chunk, dtype=float).reshape(-1, self.n_channels)
        if self.kind == "none" or len(x) == 0:
            return x.copy()
        if self._zi is None:
            self._zi = self._initial_state(x[0])
        if self._sos is not None:
            y, self._zi = signal.sosfilt(self._sos, x, axis=0, zi=self._zi)
        else:
            y, self._zi = signal.lfilter(self._fir, 1.0, x, axis=0, zi=self._zi)
        return y

    def frequency_response(self, n: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """Return (freqs_hz, magnitude) of the filter."""
        if self.kind == "none":
            f = np.linspace(0, self.fs / 2, n)
            return f, np.ones_like(f)
        if self._sos is not None:
            f, h = signal.sosfreqz(self._sos, worN=n, fs=self.fs)
        else:
            f, h = signal.freqz(self._fir, 1.0, worN=n, fs=self.fs)
        return f, np.abs(h)


def spectrum(x: np.ndarray, fs: float, remove_mean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of a 1-D signal with a Hann window.

    Scaled so that a sine of amplitude A (exactly on an FFT bin) shows a peak of A.
    Returns (freqs_hz, amplitude).
    """
    x = np.asarray(x, dtype=float)
    if remove_mean:
        x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    amp = np.abs(X) * 2.0 / w.sum()
    amp[0] /= 2.0  # DC is not doubled
    if len(x) % 2 == 0:
        amp[-1] /= 2.0  # neither is the Nyquist bin
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return freqs, amp


def to_db(amp: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Amplitude to dB (20*log10), with a floor to avoid log(0)."""
    return 20.0 * np.log10(np.maximum(amp, floor))


def peak_frequency(freqs: np.ndarray, amp: np.ndarray, fmin: float = 0.0) -> float:
    """Frequency of the largest peak above ``fmin``, refined by parabolic interpolation."""
    mask = freqs >= fmin
    if not mask.any():
        return float("nan")
    idx = np.flatnonzero(mask)
    k = idx[np.argmax(amp[idx])]
    if 0 < k < len(amp) - 1:
        a, b, c = to_db(amp[k - 1 : k + 2])
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        offset = 0.0
    df = freqs[1] - freqs[0]
    return float(freqs[k] + offset * df)
