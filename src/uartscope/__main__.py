"""Command-line entry point: python -m uartscope ..."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .dsp import FILTER_KINDS


def _parse_cutoff(text: Optional[str]):
    if text is None:
        return None
    parts = [float(p) for p in text.split(",")]
    return parts[0] if len(parts) == 1 else tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="uartscope",
        description="Real-time serial plotter with live FFT, digital filters and CSV export.",
    )
    src = ap.add_argument_group("source")
    src.add_argument("--port", help="serial port, e.g. /dev/ttyUSB0, /dev/cu.usbmodem1101, COM3")
    src.add_argument("--baud", type=int, default=115200, help="baud rate (default 115200)")
    src.add_argument("--sim", action="store_true", help="use the built-in signal simulator instead of a port")
    src.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    src.add_argument("--fs", type=float, default=1000.0, help="sample rate of the device in Hz (default 1000)")

    sim = ap.add_argument_group("simulator")
    sim.add_argument("--sim-freqs", default="10,50", help="tone frequencies in Hz (default 10,50)")
    sim.add_argument("--sim-amps", default="1,0.5", help="tone amplitudes (default 1,0.5)")
    sim.add_argument("--sim-noise", type=float, default=0.2, help="noise standard deviation (default 0.2)")
    sim.add_argument("--channels", type=int, default=1, help="number of simulated channels (default 1)")

    flt = ap.add_argument_group("filter")
    flt.add_argument("--filter", choices=FILTER_KINDS, default="none", help="filter type (default none)")
    flt.add_argument("--cutoff", help="cutoff in Hz: '30' or '5,40' for band filters")
    flt.add_argument("--method", choices=("butter", "fir"), default="butter", help="IIR Butterworth or FIR")
    flt.add_argument("--order", type=int, default=4, help="Butterworth order (default 4)")
    flt.add_argument("--numtaps", type=int, default=101, help="FIR length (default 101)")
    flt.add_argument("--q", type=float, default=30.0, help="notch quality factor (default 30)")

    view = ap.add_argument_group("view / output")
    view.add_argument("--window", type=float, default=2.0, help="seconds shown in the time plot (default 2)")
    view.add_argument("--fft-size", type=int, default=2048, help="FFT length in samples (default 2048)")
    view.add_argument("--fft-channel", type=int, default=0, help="channel shown in the FFT (default 0)")
    view.add_argument("--csv", help="record raw + filtered samples to this CSV file")
    view.add_argument(
        "--snapshot",
        help="no window: run the simulator for --window seconds of data and save the plot to this PNG",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_ports:
        from .sources import list_ports

        ports = list_ports()
        for dev, desc in ports:
            print(f"{dev}\t{desc}")
        if not ports:
            print("no serial ports found")
        return 0

    if args.snapshot:
        import matplotlib

        matplotlib.use("Agg")
        args.sim = True

    if not args.sim and not args.port:
        print("error: give --port PORT or --sim (use --list-ports to see ports)", file=sys.stderr)
        return 2

    filt_kwargs = dict(
        kind=args.filter,
        cutoff=_parse_cutoff(args.cutoff),
        order=args.order,
        method=args.method,
        numtaps=args.numtaps,
        q=args.q,
    )

    from .sources import SerialSource, SimSource

    if args.sim:
        freqs = [float(f) for f in args.sim_freqs.split(",")]
        amps = [float(a) for a in args.sim_amps.split(",")]
        source = SimSource(
            fs=args.fs,
            freqs=freqs,
            amps=amps,
            noise=args.sim_noise,
            n_channels=args.channels,
            realtime=not args.snapshot,
            chunk=max(int(args.fs / 20), 1),
            seed=0 if args.snapshot else None,
        )
        n_channels = args.channels
        label = "simulator"
    else:
        source = SerialSource(args.port, args.baud)
        n_channels = None  # known after the first valid line
        label = f"{args.port} @ {args.baud}"

    from .pipeline import Pipeline
    from .plot import Scope

    try:
        pipe = Pipeline(source, args.fs, args.window, filt_kwargs, args.csv, n_channels,
                        min_samples=args.fft_size)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        source.close()
        return 2

    filt_txt = args.filter if args.filter == "none" else f"{args.filter} {args.cutoff} Hz ({args.method})"
    scope = Scope(pipe, args.fft_size, args.fft_channel, title=f"{label}  |  filter: {filt_txt}")
    try:
        if args.snapshot:
            while pipe.raw is None or pipe.raw.total < pipe.capacity:
                pipe.step()
            scope.save(args.snapshot)
            print(f"saved {args.snapshot}")
        else:
            scope.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()
        if args.csv and pipe.recorder is not None:
            print(f"wrote {pipe.recorder.rows} samples to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
