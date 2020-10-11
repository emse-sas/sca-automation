import functools
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from lib.aes import BLOCK_LEN
from lib.cpa import COUNT_HYP, Handler
from ui.actions import timed, DEFAULT_DATA_PATH

plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["figure.titlesize"] = "x-large"


def presented(title, xlabel, ylabel):
    """Performs plot methods and formatting.

    Parameters
    ----------
    title : str
        Title of the plot.
    xlabel : str
        X-axis legend.
    ylabel : str
        Y-axis legend.
    filepath : str
        Path to save plot image.

    Returns
    -------
    function
        Decorated method.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            fig, ax = plt.gcf(), plt.gca()
            result = function(*args, **kwargs)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            fig.suptitle(title)
            return result

        return wrapper

    return decorator


def annotate(ax, annotation):
    return ax.annotate(annotation,
                xy=(.01, .95), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top')


def raw(ax, traces, limit=16, chunk=None):
    chunk = (chunk or 0) + 1
    ax.set(xlabel="Time Samples", ylabel="Hamming Weights")
    return [ax.plot(trace, label=f"iteration {d * chunk}") for d, trace in enumerate(traces[:limit])]


def mean(ax, trace):
    ax.set(xlabel="Time Samples", ylabel="Hamming Weights")
    return ax.plot(trace, color="grey", label="Temporal average")


def fft(ax, freq, spectrum, f):
    ax.set(xlabel="Frequency (MHz)", ylabel="Hamming Weight")
    return ax.plot(freq[f], spectrum[f], color="red", label="Spectrum average")


def iterations(ax, scale, guess, key, maxs):
    _, n = maxs.shape
    ax.set(xlabel="Traces acquired", ylabel="Pearson Correlation")
    scale = scale[:n]
    for h in range(COUNT_HYP):
        if h == key and h == guess:
            ax.plot(scale, maxs[h], color="r", zorder=10)
        elif h == key:
            ax.plot(scale, maxs[h], color="b", zorder=10)
        elif h == guess:
            ax.plot(scale, maxs[h], color="c", zorder=10)
        else:
            ax.plot(scale, maxs[h], color="grey")


@presented("Temporal correlation", "Time Samples", "Pearson Correlation")
def temporal(ax, cor_guess, cor_key, guess, key, exact):
    ax.set(xlabel="Time Samples", ylabel="Pearson Correlation")
    if exact:
        ax.plot(cor_key, color="r", label=f"key 0x{key:02x}")
    else:
        ax.plot(cor_guess, color="c", label=f"guess 0x{guess:02x}")
        ax.plot(cor_key, color="b", label=f"key 0x{key:02x}")


@timed("plotting data", "plot successful!")
def acquisition(traces, trace, spectrum, meta, request, path=DEFAULT_DATA_PATH, limit=16):
    """Process acquisition data, plots and saves images.

    Parameters
    ----------
    leak : sca-automation.lib.log.Leak
        Leakage data.
    request : sca-automation.lib.Request
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
    """
    meta = meta or request
    annotation = f"{'samples':<16}{traces.shape[1]}\n" \
                 f"{request}\n" \
                 f"{meta}"

    gs_kw = dict(left=0.2, hspace=0.2)
    fig, ax = plt.subplots(constrained_layout=False, gridspec_kw=gs_kw)
    raw(ax, traces, limit=limit)
    annotate(ax, annotation)
    fig.suptitle("Raw power consumptions")
    fig.legend()
    fig.savefig(os.path.join(path, request.filename("raw")))
    plt.close(fig)
    """
    annotation = f"{'samples':<16}{traces.shape[1]}\n" \
                 f"{request}\n" \
                 f"{meta}"

    gs_kw = dict(left=0.2, hspace=0.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, gridspec_kw=gs_kw)
    freq = np.fft.fftfreq(spectrum.size, 1.0 / 200e6)[:spectrum.size // 2] / 1e6
    mean(ax1, trace)
    fft(ax2, freq, spectrum, np.argsort(freq))
    annotate(ax1, annotation)
    fig.suptitle("Filtered power consumptions")
    fig.legend()
    fig.savefig(os.path.join(path, request.filename("average")))
    plt.close(fig)


class Correlation:
    scale = []


@timed("plotting data", "plot successful!")
def correlations(cor, key, request, maxs, handler, path=DEFAULT_DATA_PATH):
    """Plots temporal correlations and save images.

    Parameters
    ----------
    handler : sca-automation.lib.cpa.Handler
        Initialized handler.
    request : sca-automation.lib.Request
        Acquisition request.
    meta : sca-automation.lib.log.Meta
        Meta-data.
    path : str
        Images saving path.

    """
    gs, mx, ex = Handler.guess_stats(cor, key)
    cor_max, cor_min = Handler.guess_envelope(cor)
    maxs.append(mx)
    maxs = np.moveaxis(np.array(maxs), (0, 1, 2, 3), (3, 0, 1, 2))
    Correlation.scale.append(handler.iterations)
    """
    for i, j in product(range(BLOCK_LEN), range(BLOCK_LEN)):
        b = i * BLOCK_LEN + j
        annotation = f"imported: {handler.iterations}\n" \
                     f"guess correlation: {100 * mx[i, j, gs[i, j]]:.2f}%\n" \
                     f"key correlation: {100 * mx[i, j, key[i, j]]:.2f}%\n" \
                     f"{request}"
        gs_kw = dict(left=0.2, hspace=0.2)
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, gridspec_kw=gs_kw)
        ax1.set_xlim([Correlation.scale[0], request.iterations * request.chunks])
        iterations(ax1, Correlation.scale, gs[i, j], key[i, j], maxs[i, j])
        ax2.fill_between(range(cor.shape[3]), cor_max[i, j], cor_min[i, j], color="grey")
        temporal(ax2, cor[i, j, gs[i, j]], cor[i, j, key[i, j]], gs[i, j], key[i, j], ex[i, j])
        annotate(ax1, annotation)
        fig.suptitle(f"Correlation byte {b}")
        fig.legend()
        fig.savefig(os.path.join(path, request.filename("cor", f"_b{b}")))

        plt.close(fig)
    """

    print(f"exact guess: {np.count_nonzero(ex)}/{BLOCK_LEN * BLOCK_LEN}\n{ex}")
    print(f"key:\n{key}")
    print(f"guess:\n{gs}")
