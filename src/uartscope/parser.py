"""Parse text lines from a serial device into numeric samples.

Accepted line formats (one sample per line, one value per channel):

    1.23
    1.23,4.56
    1.23 4.56        (spaces, tabs or semicolons also work)
    a:1.23,b:4.56    (Arduino Serial Plotter style labels are ignored)
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

_SPLIT = re.compile(r"[,;\s]+")


def parse_line(line: Union[str, bytes], n_channels: Optional[int] = None) -> Optional[List[float]]:
    """Return the values in ``line`` as floats, or ``None`` if the line is not valid.

    If ``n_channels`` is given, lines with a different number of values are rejected.
    """
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="ignore")
    tokens = [t for t in _SPLIT.split(line.strip()) if t]
    if not tokens:
        return None
    values = []
    for tok in tokens:
        if ":" in tok:
            tok = tok.rsplit(":", 1)[1]
        try:
            values.append(float(tok))
        except ValueError:
            return None
    if n_channels is not None and len(values) != n_channels:
        return None
    return values


class LineAssembler:
    """Collect raw byte chunks and return complete lines.

    Serial reads arrive in arbitrary pieces; a line may be split across reads.
    """

    def __init__(self, max_line: int = 4096):
        self._buf = b""
        self._max_line = max_line

    def feed(self, data: bytes) -> List[bytes]:
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        if len(self._buf) > self._max_line:  # no newline for too long: garbage, drop it
            self._buf = b""
        return lines
