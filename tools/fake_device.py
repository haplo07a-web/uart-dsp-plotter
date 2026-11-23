"""Virtual serial device for testing without hardware (Linux / macOS only).

Creates a pseudo-terminal pair and streams simulated samples as text lines,
exactly like a microcontroller doing Serial.println(value).

    python tools/fake_device.py --fs 500
    # prints e.g.  port: /dev/ttys004
    python -m uartscope --port /dev/ttys004 --fs 500 --filter lowpass --cutoff 20
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uartscope.sources import SimSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=float, default=500.0, help="sample rate in Hz (default 500)")
    ap.add_argument("--freqs", default="10,50", help="tone frequencies in Hz (default 10,50)")
    ap.add_argument("--amps", default="1,0.5", help="tone amplitudes (default 1,0.5)")
    ap.add_argument("--noise", type=float, default=0.2, help="noise standard deviation (default 0.2)")
    ap.add_argument("--channels", type=int, default=1, help="values per line (default 1)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, signal.default_int_handler)  # `kill` stops cleanly, like Ctrl+C
    master, slave = os.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    print(f"port: {os.ttyname(slave)}   (Ctrl+C to stop)", flush=True)

    src = SimSource(
        fs=args.fs,
        freqs=[float(f) for f in args.freqs.split(",")],
        amps=[float(a) for a in args.amps.split(",")],
        noise=args.noise,
        n_channels=args.channels,
        realtime=True,
    )
    try:
        while True:
            chunk = src.read()
            if len(chunk):
                text = "".join(",".join(f"{v:.4f}" for v in row) + "\n" for row in chunk)
                try:
                    os.write(master, text.encode())
                except BlockingIOError:
                    pass  # nobody is reading; drop data like a real UART would
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        os.close(master)
        os.close(slave)
    return 0


if __name__ == "__main__":
    sys.exit(main())
