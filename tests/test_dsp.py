import numpy as np
import pytest
from scipy import signal

from uartscope.dsp import StreamingFilter, peak_frequency, spectrum

FS = 1000.0


def tone(f, n, fs=FS, amp=1.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f * t)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="lowpass", cutoff=30.0, method="butter"),
        dict(kind="bandpass", cutoff=(5.0, 40.0), method="butter"),
        dict(kind="highpass", cutoff=20.0, method="fir", numtaps=81),
        dict(kind="notch", cutoff=50.0),
    ],
)
def test_streaming_equals_batch(kwargs):
    """Filtering in random-size chunks must equal filtering everything at once."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3000, 2))
    whole = StreamingFilter(fs=FS, n_channels=2, **kwargs).process(x)

    f = StreamingFilter(fs=FS, n_channels=2, **kwargs)
    cuts = np.sort(rng.choice(np.arange(1, len(x)), size=40, replace=False))
    parts = [f.process(c) for c in np.split(x, cuts)]
    np.testing.assert_allclose(np.vstack(parts), whole, atol=1e-10)


def test_batch_matches_scipy_reference():
    x = np.random.default_rng(2).standard_normal(2000)
    f = StreamingFilter("lowpass", FS, 30.0, order=4, steady_start=False)
    sos = signal.butter(4, 30.0, fs=FS, output="sos")
    np.testing.assert_allclose(f.process(x).ravel(), signal.sosfilt(sos, x), atol=1e-12)


def test_lowpass_removes_high_tone():
    n = 4000
    x = tone(10, n) + tone(200, n)
    y = StreamingFilter("lowpass", FS, 40.0, order=6).process(x).ravel()[1000:]
    ref = tone(10, n)[1000:]
    assert abs(rms(y) - rms(ref)) < 0.05  # the 10 Hz part survives, the 200 Hz part is gone


def test_notch_kills_mains_hum():
    n = 8000
    y = StreamingFilter("notch", FS, 50.0, q=30).process(tone(50, n)).ravel()
    assert rms(y[4000:]) < 0.02


@pytest.mark.parametrize("method", ["butter", "fir"])
def test_steady_start_no_jump_on_dc(method):
    """A constant input (different per channel) must pass through a lowpass with no start transient."""
    x = np.tile([3.3, -1.2], (500, 1))
    y = StreamingFilter("lowpass", FS, 10.0, method=method, n_channels=2).process(x)
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_bad_arguments():
    with pytest.raises(ValueError):
        StreamingFilter("lowpass", FS, 600.0)  # above Nyquist
    with pytest.raises(ValueError):
        StreamingFilter("bandpass", FS, (40.0, 5.0))
    with pytest.raises(ValueError):
        StreamingFilter("wobble", FS, 10.0)
    with pytest.raises(ValueError):
        StreamingFilter("lowpass", FS, None)


def test_spectrum_amplitude_on_bin():
    n = 2048
    f0 = 64 * FS / n  # exactly on bin 64
    freqs, amp = spectrum(tone(f0, n, amp=2.5), FS)
    k = np.argmax(amp)
    assert freqs[k] == pytest.approx(f0)
    assert amp[k] == pytest.approx(2.5, rel=1e-6)


def test_peak_frequency_between_bins():
    n = 4096
    f0 = 123.4
    freqs, amp = spectrum(tone(f0, n), FS)
    assert peak_frequency(freqs, amp, fmin=1.0) == pytest.approx(f0, abs=0.05)
