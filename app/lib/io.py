import os

import serial


def _read_serial(ser, terminator=b"\n", n=16):
    """Reads data from serial port.

    Parameters
    ----------
    ser: serial.Serial
        Serial port to read.
    terminator: bytes, optional
        End bytes sequence.
    n: int, optional
        Count of chars to scan from the end to find end sequence.
    Returns
    -------
    bytes
        Binary data from the serial port.

    """
    s = bytearray(ser.read(n))
    while s[-n:].find(terminator) == -1:
        while ser.in_waiting == 0:
            continue
        while ser.in_waiting != 0:
            s += ser.read_all()
    return bytes(s)


def _write_serial(s, ser, flush=False):
    if flush:
        ser.flush()
    ser.write(s)
    return ser


def read_file(path):
    with open(path, "rb") as file:
        s = file.read()
    return s


def write_file(path, s):
    with open(path, "wb+") as file:
        file.write(s)
    return s


def acquire_serial(port, cmd, baud=921_600, terminator=b"\n"):
    with serial.Serial(port, baud, parity=serial.PARITY_NONE, xonxoff=False) as ser:
        _write_serial(f"{cmd}\n".encode(), ser, flush=True)
        s = _read_serial(ser, terminator=terminator)
    return s


def acquire_chunks(port, cmd, count, process, prepare=None, baud=921_600, terminator=b"\n"):
    prepare = prepare or (lambda c: c)
    with serial.Serial(port, baud, parity=serial.PARITY_NONE, xonxoff=False) as ser:
        for chunk in range(count):
            prepare(chunk)
            _write_serial(f"{cmd}\n".encode(), ser, flush=True)
            s = _read_serial(ser, terminator=terminator)
            process(s, chunk)
