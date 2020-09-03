"""High-level API module.

This module provides core features to ease the writing
of an application based on the library :

* Display application status
* Measure timing performance
* Acquire, import and export data
* Perform data filtering and plots
* Perform files creation and deletion routines

"""

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, signal

from lib import data, aes, cpa, utils, io
from lib import traces as tr
from lib.data import Keywords, Request

MODES = ["hw", "sw"]  # available encryption sources
SOURCES = ["serial", "file"]
F_SAMPLING = 200e6  # sensors sampling frequency
DEFAULT_DIR = os.path.join("..", "data")
ACQ_DIR = "acquisition"  # label for the acquisition directory
COR_DIR = "correlation"  # label for the correlation  directory
DATA_PATH = os.path.join("..", "data")
MEDIA_PATH = os.path.join("..", "media")
IMG_PATH = os.path.join(MEDIA_PATH, "img")
DATA_PATH_ACQ = os.path.join(DATA_PATH, ACQ_DIR)
DATA_PATH_COR = os.path.join(DATA_PATH, COR_DIR)
IMG_PATH_ACQ = os.path.join(IMG_PATH, ACQ_DIR)
IMG_PATH_COR = os.path.join(IMG_PATH, COR_DIR)


def init(request, path=DEFAULT_DIR):
    count = 0
    loadpath = None
    if request.source == Request.Sources.FILE:
        loadpath = path
        path = os.sep.join(path.split(os.sep)[:-1])
    while True:
        try:
            savepath = os.path.join(path, f"{count}")
            os.mkdir(savepath)
            break
        except FileExistsError:
            count += 1
    return savepath, loadpath or savepath


@utils.operation_decorator("acquiring bytes", "\nacquisition successful!")
def acquire(request, process, prepare=None, path=DEFAULT_DIR):
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
    request : sca-automation.core.Request
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


@utils.operation_decorator("saving data", "export successful!")
def save(request, s=None, leak=None, channel=None, meta=None, chunk=None, path=DEFAULT_DIR):
    """Exports CSV data to CSV files.

    If ``iterations`` and ``mode`` are not specified
    ``meta`` must be given.

    Otherwise these parameters must represent the expected values
    received by the SoC in order to track unexpected behavior.

    Parameters
    ----------
    request : sca-automation.core.Request
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
        io.write_file(os.path.join(path, request.filename(suffix=f"_{chunk}.bin" if chunk else ".bin")), s)
    if channel:
        channel.write_csv(os.path.join(path, request.filename("channel", ".csv")), append)
    if leak:
        leak.write_csv(os.path.join(path, request.filename("leak", ".csv")), append)
    if meta:
        meta.write_csv(os.path.join(path, request.filename("meta", ".csv")), append)


@utils.operation_decorator("loading data", "import successful!")
def load(request, callback=None, path=None):
    """Imports CSV files and parse data.

    Parameters
    ----------
    request : sca-automation.core.Request
        Acquisition request.
    path : str, optional
        Path of CSV files.
    callback : function, optional
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
    path = path or os.path.join(DATA_PATH_ACQ, request.mode)
    meta = data.Meta(os.path.join(path, request.filename("meta", ".csv")))
    count = request.iterations
    if request.chunks:
        for chunk in range(request.chunks):
            start = chunk * count
            channel = data.Channel(os.path.join(path, request.filename("channel", ".csv")), count, start)
            leak = data.Leak(os.path.join(path, request.filename("leak", ".csv")), count, start)
            callback(channel, leak, meta, chunk)
        return
    channel = data.Channel(os.path.join(path, request.filename("channel", ".csv")))
    leak = data.Leak(os.path.join(path, request.filename("leak", ".csv")))
    callback(channel, leak, meta, None)


@utils.operation_decorator("processing traces", "processing successful!")
def filter_traces(leak):
    """Filters raw traces to ease correlation.

    Parameters
    ----------
    leak : sca-automation.lib.log.Leak
        Leakage data.

    Returns
    -------
    traces : np.ndarray
        Filtered traces matrix.
    b : np.ndarray
        Filter's denominator coefficients.
    a : np.ndarray
        Filter's numerator coefficients.

    """
    traces = np.array(tr.crop(leak.traces))
    f_c = 13e6
    order = 4
    w = f_c / (F_SAMPLING / 2)
    b, a, *_ = signal.butter(order, w, btype="highpass", output="ba")

    for trace in traces:
        trace[:] = signal.filtfilt(b, a, trace)

    return traces, b, a


@utils.operation_decorator("creating handler", "handler successfully create!")
def update_handler(channel, traces, model, handler=None):
    """Creates a correlation handler.

    Parameters
    ----------
    channel : sca-automation.lib.log.Data
        Encryption data.
    traces : np.ndarray
        Traces matrix.
    model : int
        Model index.

    Returns
    -------
    sca-automation.lib.cpa.Handler
        Handler initialized to perform correlation over ``traces``.


    """

    key = aes.words_to_block(channel.keys[0])
    if model == cpa.Models.SBOX:
        blocks = [aes.words_to_block(block) for block in channel.plains]
    elif model == cpa.Models.INV_SBOX:
        key = aes.key_expansion(key)[10].T
        blocks = [aes.words_to_block(block) for block in channel.ciphers]
    else:
        return
    if handler:
        return handler.accumulate(blocks, traces), key
    return cpa.Handler(np.array(blocks), traces, model=model), key


def update_sum(x, traces):
    if x is None:
        x = np.zeros((traces.shape[1]))
    x += np.sum(traces, axis=0)
    return x


@utils.operation_decorator("plotting data", "plot successful!")
def plot_acq(traces, mean, spectrum, meta, request, path=DEFAULT_DIR, limit=16):
    """Process acquisition data, plots and saves images.

    Parameters
    ----------
    leak : sca-automation.lib.log.Leak
        Leakage data.
    request : sca-automation.core.Request
        Acquisition request.
        Encryption mode.
    meta : sca-automation.lib.log.Meta
        Meta-data.
    path : str
        Images saving path.
    limit : int
        Count of raw acquisition curves to plot.

    Returns
    -------
    traces : np.ndarray
        Traces matrix.
    mean : np.ndarray
        Average trace.
    spectrum : np.ndarray
        Spectrum of the average.
    freq : np.ndarray
        Spectrum's frequencies.

    """

    order = 4
    w = 5e6 / (F_SAMPLING / 2)
    b, a, *_ = signal.butter(order, w, btype="highpass", output="ba")
    filtered = signal.filtfilt(b, a, mean)

    order = 4
    w0 = 49e6 / (F_SAMPLING / 2)
    w1 = 51e6 / (F_SAMPLING / 2)
    b, a, *_ = signal.butter(order, [w0, w1], btype="bandstop", output="ba")
    filtered = signal.filtfilt(b, a, filtered)
    freq = np.fft.fftfreq(spectrum.size, 1.0 / F_SAMPLING)
    n, m = traces.shape
    freq = freq[:spectrum.size // 2] / 1e6
    f = np.argsort(freq)

    meta = meta or request
    infos = f"(iterations: {meta.iterations}, samples: {m}, sensors: {meta.sensors})"
    plt.rcParams["figure.figsize"] = (16, 9)

    @utils.plot_decorator(f"Raw power consumptions {infos}",
                          "Time Samples", "Hamming Weights",
                          os.path.join(path, request.filename("sca_raw")))
    def plot_raw():
        return [plt.plot(traces[d], label=f"sample {d}") for d in range(0, limit)]

    @utils.plot_decorator(f"Average power consumption {infos}",
                          "Time Samples", "Hamming Weights",
                          os.path.join(path, request.filename("sca_avg")))
    def plot_mean():
        return plt.plot(mean, color="grey", label="Raw signal")

    @utils.plot_decorator(f"Average power consumption {infos}",
                          "Frequency (MHz)", "Hamming Weight",
                          os.path.join(path, request.filename("sca_fft")))
    def plot_fft():
        return plt.plot(freq[f], spectrum[f], color="red", label="Raw spectrum")

    plot_raw()
    plot_mean()
    plot_fft()

    return traces, mean, spectrum, freq


@utils.operation_decorator("plotting data", "plot successful!")
def plot_cor(handler, key, request, meta=None, path=None):
    """Plots temporal correlations and save images.

    Parameters
    ----------
    handler : sca-automation.lib.cpa.Handler
        Initialized handler.
    request : sca-automation.core.Request
        Acquisition request.
    meta : sca-automation.lib.log.Meta
        Meta-data.
    path : str
        Images saving path.

    """
    cor = handler.correlations()
    guess, maxs, exact = cpa.Handler.guess_stats(cor, key)
    cor_max, cor_min = handler.guess_envelope(cor)
    _, _, m = cor_max.shape

    meta = meta or request
    path = path or os.path.join(IMG_PATH_COR, request.mode)
    plt.rcParams["figure.figsize"] = (16, 9)

    for i, j in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN)):
        b = i * aes.BLOCK_LEN + j
        g = 100 * maxs[i, j, guess[i, j]]
        k = 100 * maxs[i, j, key[i, j]]
        infos = f"(iterations: {meta.iterations}, guess correlation: {g:.2f}%, key correlation:{k:.2f}%)"
        plt.fill_between(range(m), cor_max[i, j], cor_min[i, j], color="grey")

        @utils.plot_decorator(
            f"Correlation byte {b} {infos}",
            "Time Samples",
            "Pearson Correlation",
            os.path.join(path, request.filename("sca_cor", f"_b{b}")))
        def plot_guess():
            if exact[i, j]:
                plt.plot(cor[i, j, key[i, j]], color="r",
                         label=f"key 0x{key[i, j]:02x}")
            else:
                plt.plot(cor[i, j, key[i, j]], color="b",
                         label=f"key 0x{key[i, j]:02x}")
                plt.plot(cor[i, j, guess[i, j]], color="c",
                         label=f"guess 0x{guess[i, j]:02x}")

        plot_guess()

    print(
        f"exact guess: {np.count_nonzero(exact)}/{aes.BLOCK_LEN * aes.BLOCK_LEN}\n{exact}")
    print(f"key:\n{key}")
    print(f"guess:\n{guess}")


@utils.operation_decorator("removing logs", "remove success!")
def remove_logs():
    """Removes all the log files, CSV and binary.

    """
    utils.remove_subdir_files(DATA_PATH_ACQ)


@utils.operation_decorator("removing acquisition images", "remove success!")
def remove_acquisition_images():
    """Removes all the acquisition images.

    """
    utils.remove_subdir_files(IMG_PATH_ACQ)


@utils.operation_decorator("removing correlation images", "remove success!")
def remove_correlation_images():
    """Removes all the correlations images.

    """
    utils.remove_subdir_files(IMG_PATH_COR)


@utils.operation_decorator("creating log dirs", "create success!")
def create_logs_dir():
    """Creates data directories.

    """
    utils.try_create_dir(DATA_PATH)
    utils.create_subdir(MODES, DATA_PATH_ACQ)
    utils.create_subdir(MODES, DATA_PATH_COR)


@utils.operation_decorator("creating images dirs", "create success!")
def create_images_dir():
    """Creates images directories.

    """
    utils.try_create_dir(MEDIA_PATH)
    utils.try_create_dir(IMG_PATH)
    utils.create_subdir(MODES, IMG_PATH_ACQ)
    utils.create_subdir(MODES, IMG_PATH_COR)
