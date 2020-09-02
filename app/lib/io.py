import os

import serial


def read_serial(ser, terminator=b"\n", n=8):
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


def write_serial(s, ser, flush=False):
    if flush:
        ser.flush()
    ser.write(s)
    return ser


def acquire_serial(port, cmd, baud=921_600, terminator=b"\n", path=None, write=True):
    path = path or port
    with serial.Serial(port, baud, parity=serial.PARITY_NONE, xonxoff=False) as ser:
        write_serial(f"{cmd}\n".encode(), ser, flush=True)
        s = read_serial(ser, terminator=terminator)
    if not write:
        return s
    with open(path, "wb+") as file:
        file.write(s)
    return s


def acquire_chunks(port, cmd, callback, count=1, baud=921_600, terminator=b"\n", path=None, write=True):
    path = path or port
    with serial.Serial(port, baud, parity=serial.PARITY_NONE, xonxoff=False) as ser:
        for chunk in range(count):
            write_serial(f"{cmd}\n".encode(), ser, flush=True)
            s = read_serial(ser, terminator=terminator)
            callback(s, chunk)
            if not write:
                return s
            split = os.path.splitext(path)
            with open(f"{split[0]}_{chunk}{split[1]}", "wb+") as file:
                file.write(s)


def read_file(path):
    with open(path, "rb") as file:
        s = file.read()
    return s
