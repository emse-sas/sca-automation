"""High-level API module.

This module provides core features to ease the writing
of an application based on the library :

* Display application status
* Measure timing performance
* Acquire, import and export data
* Perform data filtering and plots
* Perform files creation and deletion routines

"""
import functools
import os
import time
from datetime import timedelta
from lib import data, io
from lib.data import Keywords, Request

DEFAULT_DATA_DIR = "data"
DEFAULT_DATA_PATH = os.path.join("..", DEFAULT_DATA_DIR)


def timed(title, message=None):
    """Executes a function and prints messages and duration.

    Parameters
    ----------
    title : str
        Starting message.

    message : str
        End message on success.
    Returns
    -------
        function
            Decorated method.

    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            print(f"\n*** {title} ***")
            t_start = time.perf_counter()
            result = function(*args, **kwargs)
            t_end = time.perf_counter()
            if message:
                print(f"{message}\n{'elapsed':<16} {timedelta(seconds=t_end - t_start)}")
            return result

        return wrapper

    return decorator


def sizeof(num, suffix="B"):
    """Converts and format a number to a file size unit.

    Parameters
    ----------
    num : int
        Number to format.
    suffix : str
        Unit suffix.
    Returns
    -------
    str
        Formatted number.
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}{'Yi'}{suffix}"


def init(request, path=DEFAULT_DATA_PATH):
    count = 0
    loadpath = None
    if request.source == Request.Sources.FILE:
        loadpath = path
        path = os.sep.join(path.split(os.sep)[:-1])
    while True:
        savepath = os.path.join(path, f"{count}")
        try:
            os.mkdir(savepath)
            break
        except FileExistsError:
            count += 1
        except FileNotFoundError:
            os.mkdir(savepath.split(os.sep)[-1])

    return savepath, loadpath or savepath


@timed("acquiring bytes", "\nacquisition successful!")
def acquire(request, process, prepare=None, path=DEFAULT_DATA_PATH):
    """Acquires binary data from serial or file.

    If ``sources`` is a serial channel such as ``COM1``,
    the acquired data is write-back to a log file.
    ``path`` must be the path of the write-back file.

    Otherwise, it reads the file with the prefix given by ``sources``.
    Therefore ``path`` must be the path of the file to read.

    Notes
    -----

    * The default path is the data path created by ``setup.py``.

    Parameters
    ----------
    request : sca-automation.lib.Request
        Acquisition request.

    path : str, optional
        Export or sources path.
    Returns
    -------
    bytes
        Binary data string if no callback is provided else None

    """
    cmd = request.command(Request.ACQ_CMD_NAME)
    terminator = Keywords.END_ACQ_TAG
    filepath = os.path.join(path, request.filename(suffix=".bin"))
    if request.source == Request.Sources.SERIAL:
        if request.chunks:
            io.acquire_chunks(request.name, cmd, request.chunks, process, prepare, terminator=terminator)
        else:
            prepare(None, None)
            s = io.acquire_serial(request.name, cmd, terminator=terminator)
            process(s, None)
    elif request.source == Request.Sources.FILE:
        if request.chunks:
            for chunk in range(request.chunks):
                filepath = os.path.join(path, request.filename(request.name, f"_{chunk}.bin"))
                prepare(None, chunk)
                s = io.read_file(filepath)
                process(s, chunk)
        else:
            prepare(None, None)
            s = io.read_file(filepath)
            process(s, None)
    else:
        raise ValueError(f"unrecognized request source: {request.source}")


@timed("saving data", "export successful!")
def save(request, s=None, leak=None, channel=None, meta=None, noise=None, chunk=None, path=DEFAULT_DATA_PATH):
    """Exports CSV data to CSV files.

    If ``iterations`` and ``mode`` are not specified
    ``meta`` must be given.

    Otherwise these parameters must represent the expected values
    received by the SoC in order to track unexpected behavior.

    Parameters
    ----------
    request : sca-automation.lib.Request
        Acquisition request.
    s : bytes, optional
    meta : sca-automation.lib.log.Meta, optional
        Meta-data.
    leak : sca-automation.lib.log.Leak, optional
        Leakage data.
    channel : sca-automation.lib.log.Channel, optional
        Encryption data.
    path : str, optional
        Path of CSV files.

    """
    append = request.chunks is not None
    if s:
        io.write_file(os.path.join(path, request.filename(suffix=f"_{chunk}.bin" if chunk is not None else ".bin")), s)
    if channel:
        channel.write_csv(os.path.join(path, request.filename("channel", ".csv")), append)
    if leak:
        leak.write_csv(os.path.join(path, request.filename("leak", ".csv")), append)
    if meta:
        meta.write_csv(os.path.join(path, request.filename("meta", ".csv")), append)
    if noise:
        noise.write_csv(os.path.join(path, request.filename("noise", ".csv")), append)


@timed("loading data", "\nimport successful!")
def load(request, process, prepare=None, path=DEFAULT_DATA_PATH):
    """Imports CSV files and parse data.

    Parameters
    ----------
    request : sca-automation.lib.Request
        Acquisition request.
    path : str, optional
        Path of CSV files.
    process : function, optional
        Function to be executed after loading
    Returns
    -------

    channel : sca-automation.lib.log.Data
        Encryption data.
    leak : sca-automation.lib.log.Leak
        Leakage data.
    meta : sca-automation.lib.log.Meta
        Meta-data.
    """
    meta = data.Meta(os.path.join(path, request.filename("meta", ".csv")))
    count = request.iterations
    if request.chunks:
        for chunk in range(request.chunks):
            prepare(None, None, None, chunk)
            start = chunk * count
            channel = data.Channel(os.path.join(path, request.filename("channel", ".csv")), count, start)
            leak = data.Leak(os.path.join(path, request.filename("leak", ".csv")), count, start)
            process(channel, leak, meta, chunk)
        return
    prepare(None, None, None, None)
    channel = data.Channel(os.path.join(path, request.filename("channel", ".csv")))
    leak = data.Leak(os.path.join(path, request.filename("leak", ".csv")))
    process(channel, leak, meta, None)
