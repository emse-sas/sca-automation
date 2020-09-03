"""Perform signal processing on power consumption traces.

This module is designed to provide fast signal processing
function for power consumption signals.

Examples
--------
>>> from lib import data, read, traces as tr
>>> from numpy import np
>>> s = read.file("path/to/binary/file")
>>> parser = data.Parser.from_bytes(s)
>>> traces = np.array(tr.crop(parser.leak.traces))

>>> from lib import data, read, traces as tr
>>> from numpy import np
>>> s = read.file("path/to/binary/file")
>>> parser = data.Parser.from_bytes(s)
>>> traces = np.array(tr.pad(parser.leak.traces))

"""

import numpy as np
from scipy import stats


def crop(traces, end=None):
    """Crops all the traces signals to have the same duration.

    If ``end`` parameter is not provided the traces are cropped to have
    the same duration as the shortest given trace.

    Parameters
    ----------
    traces : list[list[int]]
        2D list of numbers representing the trace signal.
    end : int, optional
        Index after which the traces are truncated.
        Must be inferior to the length of the shortest trace.

    Returns
    -------
    list[list[int]]
        Cropped traces.

    """
    m = min(map(len, traces))
    m = min(end or m, m)
    return [trace[:m] for trace in traces]


def pad(traces, fill=0, end=None):
    """Pads all the traces signals have the same duration.

    If ``end`` parameter is not provided the traces are padded to have
    the same duration as the longest given trace.

    Parameters
    ----------
    traces : list[list[int]]
        2D list of numbers representing the trace signal.
    fill : int, optional
        Padding value to insert after the end of traces.
    end : int, optional
        New count of samples of the traces.
        Must be greater than the length of the longest trace.
    Returns
    -------
    list[list[int]]
        Padded traces.

    """
    samples = list(map(len, traces))
    m = max(samples)
    m = max(end or m, m)
    return [trace + [fill] * (m - read) for trace, read in zip(traces, samples)]


def adjust(traces=None, ref=None, fill=0):
    if ref is None:
        return crop(traces)
    m = len(traces)
    n = len(ref)
    if m > n:
        return crop(traces, end=n)
    elif n > m:
        return pad(traces, end=n, fill=fill)


def sync(traces, step=1, stop=None):
    """Synchronize trace signals by correlating them

    This function implements an algorithm based on Pearson's
    correlation to synchronize signals peaks.

    More precisely, it compares the traces to a reference trace
    by rolling theses forward or backward. The algorithm search
    for the roll value that maximizes pearson correlation.
    
    Parameters
    ----------
    traces : np.ndarray
        2D numbers array representing cropped or padded traces data.
    step : int, optional
        Rolling step, if equals n, the trace will be rolled
        n times in both directions at each rolling iteration.
    stop : int, optional
        Rolling stop, maximum roll to perform.

    Returns
    -------
        np.ndarray
            2D array representing synchronized traces.

    """
    ref = traces[0]
    n, m = traces.shape
    strides_pos = ref.strides * 2
    strides_neg = (-strides_pos[0], strides_pos[1])
    shape = (m, m)
    stop = min(stop or m, m)
    shifts = list(range(0, stop, step))

    for trace in traces:
        strided = np.lib.stride_tricks.as_strided(trace, shape, strides_pos)
        try:
            buffer = _pearsonr_from_ref(ref, strided, shifts)
        except ValueError:
            continue

        argmax_pos = np.int(np.argmax(buffer))
        max_pos = buffer[argmax_pos]
        strided = np.lib.stride_tricks.as_strided(trace, shape, strides_neg)
        try:
            buffer = _pearsonr_from_ref(ref, strided, shifts)
        except ValueError:
            continue
        argmax_neg = np.int(np.argmax(buffer))
        max_neg = buffer[argmax_neg]
        if max_neg < max_pos:
            trace[:] = np.roll(trace, -shifts[argmax_pos])
        else:
            trace[:] = np.roll(trace, shifts[argmax_neg])

    trace = traces[n - 1]
    shifts = list(range(-stop, stop, step))
    buffer = list(map(lambda shift: stats.pearsonr(ref, np.roll(trace, shift))[0], shifts))
    trace[:] = np.roll(trace, np.argmax(buffer) - stop)

    return traces


def _pearsonr_from_ref(ref, strided, shifts):
    return list(map(lambda shift: stats.pearsonr(ref, strided[shift])[0], shifts))
