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
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from scipy import fft, signal
from lib import log, aes, cpa, traces as tr
from lib import utils

MODES = ["hw", "sw"]  # available encryption sources
F_SAMPLING = 200e6  # sensors sampling frequency
ACQ_DIR = "acquisition"  # label for the acquisition directory
COR_DIR = "correlation"  # label for the correlation  directory
DATA_PATH = os.path.join("..", "data")
MEDIA_PATH = os.path.join("..", "media")
IMG_PATH = os.path.join(MEDIA_PATH, "img")
DATA_PATH_ACQ = os.path.join(DATA_PATH, ACQ_DIR)
DATA_PATH_COR = os.path.join(DATA_PATH, COR_DIR)
IMG_PATH_ACQ = os.path.join(IMG_PATH, ACQ_DIR)
IMG_PATH_COR = os.path.join(IMG_PATH, COR_DIR)


class Request:
    """Data processing request.

    This class provides a simple abstraction to wrap
    file naming arguments during acquisition, import or export.

    The these arguments combined together form a *data request*
    specifying all the characteristics of the target data-set.

    Attributes
    ----------
    iterations : int
        Requested count of traces.
    mode : str
        Encryption mode.
    inv : bool
        True if encryption direction is decrypt.
    serial : bool
        True if an acquisition from serial port is requested.
    """

    def __init__(self, iterations, mode, inv, serial):
        """Initializes a request with attributes.

        Attributes
        ----------
        iterations : int
            Requested count of traces.
        mode : str
            Encryption mode.
        inv : bool
            True if encryption direction is decrypt.
        serial : bool
            True if an acquisition from serial port is requested.
        """
        self.iterations = iterations
        self.mode = mode
        self.inv = inv
        self.serial = serial

    @classmethod
    def from_args(cls, args):
        """Initializes a request with command line args.

        Parameters
        ----------
        args
            Parsed arguments.

        """
        try:
            return Request(args.iterations, args.mode, args.inv, args.serial)
        except AttributeError:
            return Request(args.iterations, args.mode, 0, False)

    @classmethod
    def from_meta(cls, meta):
        """Initializes a request with command line arguments.

        Parameters
        ----------
        meta : sca-automation.lib.log.Meta
            Meta-data.

        """
        return Request(meta.iterations, meta.mode, meta.direction == "decrypt", False)

    def filename(self, prefix, suffix=""):
        """Creates a filename based on the request.

        This method allows to consistently name the files
        according to request's characteristics.

        Parameters
        ----------
        prefix : str
            Prefix of the filename.
        suffix : str, optional
            Suffix of the filename.

        Returns
        -------
        str :
            The complete filename.

        """
        iv = "_iv" if self.inv else ""
        return f"{prefix}_{self.mode}{iv}_{self.iterations}{suffix}"


@utils.operation_decorator("acquiring bytes", "acquisition successful!")
def acquire_bin(source, request, path=None):
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
    source : str
        Serial channel or file prefix.
    request : sca-automation.core.Request
        Acquisition request.
    path : str, optional
        Export or sources path.

    Returns
    -------
    bytes
        Binary data string

    """
    path = path or os.path.join(DATA_PATH_ACQ, request.mode)
    print(f"sources: {source}")
    if request.serial:
        s = log.read.serial(source, request.iterations,
                            request.mode, request.inv)
        log.write.bytes(s, os.path.join(path, request.filename("cmd", ".log")))
    else:
        s = log.read.file(os.path.join(path, request.filename(source, ".log")))
    print(f"buffer size: {utils.format_sizeof(len(s))}")
    return s


@utils.operation_decorator("parsing bytes", "parsing successful!")
def parse_bin(s, request):
    """Parses binary data.

    Parameters
    ----------
    s : bytes
        Binary data string.
    request : sca-automation.core.Request
        Acquisition request.

    Returns
    -------
    Parser
        Parser initialized with binary data.

    """
    parser = log.Parser.from_bytes(s, request.inv)
    print(f"traces parsed: {parser.meta.iterations}/{request.iterations}")
    return parser


@utils.operation_decorator("exporting data", "export successful!")
def export_csv(request, meta=None, leak=None, data=None, path=None):
    """Exports parser data to CSV files.

    If ``iterations`` and ``mode`` are not specified
    ``meta`` must be given.

    Otherwise these parameters must represent the expected values
    received by the SoC in order to track unexpected behavior.

    Parameters
    ----------
    request : sca-automation.core.Request
        Acquisition request.
    meta : sca-automation.lib.log.Meta, optional
        Meta-data.
    leak : sca-automation.lib.log.Leak, optional
        Leakage data.
    data : sca-automation.lib.log.Data, optional
        Encryption data.
    path : str, optional
        Path of CSV files.

    """
    path = path or os.path.join(DATA_PATH_ACQ, request.mode)
    if data:
        data.to_csv(os.path.join(path, request.filename("data", ".csv")))
    if leak:
        leak.to_csv(os.path.join(path, request.filename("leak", ".csv")))
    if meta:
        meta.to_csv(os.path.join(path, request.filename("meta", ".csv")))


@utils.operation_decorator("importing data", "import successful!")
def import_csv(request, path=None):
    """Imports CSV files and parse data.

    Parameters
    ----------
    request : sca-automation.core.Request
        Acquisition request.
    path : str, optional
        Path of CSV files.

    Returns
    -------

    leak : sca-automation.lib.log.Leak
        Leakage data.
    data : sca-automation.lib.log.Data
        Encryption data.
    meta : sca-automation.lib.log.Meta
        Meta-data.
    """
    path = path or os.path.join(DATA_PATH_ACQ, request.mode)
    data = log.Data.from_csv(os.path.join(
        path, request.filename("data", ".csv")))
    leak = log.Leak.from_csv(os.path.join(
        path, request.filename("leak", ".csv")))
    meta = log.Meta.from_csv(os.path.join(
        path, request.filename("meta", ".csv")))
    iterations = meta.iterations if meta else "--"
    print(f"traces imported: {iterations}/{request.iterations}")
    return leak, data, meta


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
def init_handler(data, traces, model):
    """Creates a correlation handler.

    Parameters
    ----------
    data : sca-automation.lib.log.Data
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
    key = aes.words_to_block(data.keys[0])
    if model == cpa.Models.SBOX:
        blocks = [aes.words_to_block(block) for block in data.plains]
    elif model == cpa.Models.INV_SBOX:
        key = aes.key_expansion(key)[10].T
        blocks = [aes.words_to_block(block) for block in data.ciphers]
    else:
        return
    return cpa.Handler(np.array(blocks), key, traces, model=model)


@utils.operation_decorator("plotting data", "plot successful!")
def plot_acq(leak, meta, request, path=None, limit=16):
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
    traces = np.array(tr.crop(leak.traces))
    mean = traces.mean(axis=0)
    spectrum = np.absolute(fft.fft(mean - np.mean(mean)))
    freq = np.fft.fftfreq(spectrum.size, 1.0 / F_SAMPLING)
    n, m = traces.shape
    freq = freq[:spectrum.size // 2] / 1e6
    f = np.argsort(freq)

    meta = meta or request
    path = path or os.path.join(IMG_PATH_ACQ, meta.mode)
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
def plot_cor(handler, request, meta=None, path=None):
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
    guess, maxs, exact = handler.guess_stats(cor)
    cor_max, cor_min = handler.guess_envelope(cor)
    key = handler.key
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
