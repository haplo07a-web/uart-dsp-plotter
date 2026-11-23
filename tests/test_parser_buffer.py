import numpy as np
import pytest

from uartscope.buffer import RingBuffer
from uartscope.parser import LineAssembler, parse_line


@pytest.mark.parametrize(
    "line, expected",
    [
        ("1.5", [1.5]),
        ("1.5,2.5\r\n", [1.5, 2.5]),
        ("1 2\t3", [1.0, 2.0, 3.0]),
        ("1;-2e3", [1.0, -2000.0]),
        ("a:1.0,b:2.0", [1.0, 2.0]),
        (b"3.25\n", [3.25]),
    ],
)
def test_parse_valid(line, expected):
    assert parse_line(line) == expected


@pytest.mark.parametrize("line", ["", "   ", "hello", "1.0,abc", b"\xff\xfe"])
def test_parse_invalid(line):
    assert parse_line(line) is None


def test_parse_channel_count():
    assert parse_line("1,2", n_channels=2) == [1.0, 2.0]
    assert parse_line("1,2,3", n_channels=2) is None


def test_line_assembler_split_chunks():
    la = LineAssembler()
    assert la.feed(b"1.0\n2.") == [b"1.0"]
    assert la.feed(b"5\n3.0") == [b"2.5"]
    assert la.feed(b"\n") == [b"3.0"]


def test_line_assembler_drops_garbage():
    la = LineAssembler(max_line=8)
    la.feed(b"x" * 20)
    assert la.feed(b"1\n") == [b"1"]


def test_ring_buffer_order_and_wrap():
    rb = RingBuffer(5, 1)
    rb.append(np.arange(3))
    assert rb.get().ravel().tolist() == [0, 1, 2]
    rb.append(np.arange(3, 8))
    assert rb.get().ravel().tolist() == [3, 4, 5, 6, 7]
    rb.append([8, 9])
    assert rb.get().ravel().tolist() == [5, 6, 7, 8, 9]
    assert len(rb) == 5 and rb.total == 10


def test_ring_buffer_big_chunk_multichannel():
    rb = RingBuffer(4, 2)
    data = np.arange(20).reshape(10, 2)
    rb.append(data)
    np.testing.assert_array_equal(rb.get(), data[-4:])
