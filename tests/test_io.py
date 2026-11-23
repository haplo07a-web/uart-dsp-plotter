import csv
import os
import sys
import time

import numpy as np
import pytest

from uartscope.__main__ import main
from uartscope.dsp import peak_frequency, spectrum
from uartscope.export import CsvRecorder
from uartscope.pipeline import Pipeline
from uartscope.sources import SerialSource, SimSource


def test_sim_source_shape_and_tone():
    src = SimSource(fs=1000, freqs=[37.0], amps=[1.0], noise=0.05, n_channels=2,
                    realtime=False, chunk=4096, seed=0)
    x = src.read()
    assert x.shape == (4096, 2)
    freqs, amp = spectrum(x[:, 1], 1000)
    assert peak_frequency(freqs, amp, fmin=1.0) == pytest.approx(37.0, abs=0.1)


def test_csv_recorder_roundtrip(tmp_path):
    path = tmp_path / "rec.csv"
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    with CsvRecorder(str(path), fs=100.0, n_channels=2) as rec:
        rec.write(raw, raw * 10)
        rec.write(raw, raw * 10)
    rows = list(csv.reader(open(path)))
    assert rows[0] == ["t_s", "ch0_raw", "ch0_filt", "ch1_raw", "ch1_filt"]
    assert len(rows) == 5
    assert rows[3] == ["0.020000", "1", "10", "2", "20"]


def test_pipeline_with_sim_and_csv(tmp_path):
    src = SimSource(fs=500, realtime=False, chunk=100, seed=0)
    path = tmp_path / "p.csv"
    pipe = Pipeline(src, 500, window_s=1.0, filt_kwargs=dict(kind="lowpass", cutoff=20.0),
                    csv_path=str(path), n_channels=1)
    for _ in range(8):
        pipe.step()
    pipe.close()
    assert len(pipe.raw) == 500 and pipe.raw.total == 800
    assert sum(1 for _ in open(path)) == 801  # header + samples


def test_snapshot_cli(tmp_path):
    out = tmp_path / "shot.png"
    rc = main(["--snapshot", str(out), "--fs", "500", "--filter", "bandpass", "--cutoff", "5,20",
               "--fft-size", "1024", "--window", "1"])
    assert rc == 0
    assert out.stat().st_size > 10_000


def test_cli_needs_source(capsys):
    assert main([]) == 2
    assert "--port" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="pseudo-terminals are POSIX only")
def test_serial_source_over_pty():
    """End-to-end serial path: bytes written to a pty arrive as parsed samples."""
    import tty

    master, slave = os.openpty()
    tty.setraw(slave)
    src = SerialSource(os.ttyname(slave), 115200)
    try:
        os.write(master, b"1.0,2.0\n3.0,4.0\nbad line\n5.0,")
        rows = np.empty((0, 2))
        deadline = time.monotonic() + 2.0
        while len(rows) < 2 and time.monotonic() < deadline:
            chunk = src.read()
            if len(chunk):
                rows = np.vstack([rows, chunk])
            time.sleep(0.01)
        np.testing.assert_array_equal(rows, [[1.0, 2.0], [3.0, 4.0]])
        assert src.dropped == 1
        os.write(master, b"6.0\n")
        deadline = time.monotonic() + 2.0
        chunk = src.read()
        while len(chunk) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
            chunk = src.read()
        np.testing.assert_array_equal(chunk, [[5.0, 6.0]])
    finally:
        src.close()
        os.close(master)
        os.close(slave)
