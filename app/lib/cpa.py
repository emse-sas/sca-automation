"""Numpy based module for CPA side-channel attacks.

This module is provides abstractions to handle
side channel attacks via power consumption acquisition.

It features a trace accumulator to avoid storing all the
traces in memory. It also implements a fast Pearson correlation algorithm
to retrieve attack results in a reasonable amount of time.

Examples
--------
>>> from lib import data, cpa, aes, traces as tr
>>> import numpy as np
>>> meta = data.Meta.from_csv("path/to/meta.csv")
>>> data = data.Channel.from_csv("path/to/data.csv")
>>> leak = data.Leak.from_csv("path/to/leak.csv")
>>> blocks = np.array([aes.words_to_block(block) for block in data.plains], dtype=np.uint8)
>>> key = aes.words_to_block(data.keys[0])
>>> traces = np.array(tr.crop(leak.traces))
>>> handler = cpa.Handler(blocks, key, traces)
>>> correlations = handler.correlations()

"""

from itertools import product

import numpy as np

from lib import aes

COUNT_HYP = 256  # Count of key hypothesis for one byte
COUNT_CLS = 256  # Traces with the same byte value in a given position


class Models:
    """CPA power consumption models.

    """

    SBOX = 0
    INV_SBOX = 1


class Handler:
    """CPA correlation handler interface.

    Attributes
    ----------
    blocks: np.ndarray
        Encrypted data blocks for each trace.
    key: np.ndarray
        Key data block for all the traces.
    hyp: np.ndarray
        Value of power consumption for each hypothesis and class.
    lens: np.ndarray
        Count of traces per class.
    sums: np.ndarray
        Average trace per class.
    sums2: np.ndarray
        Standard deviation trace per class.
    sum: np.ndarray
        Average trace for all classes.
    sum2: np.ndarray
        Standard deviation trace for all classes.

    """

    def __init__(self, blocks, traces, model=Models.SBOX):
        """Allocates memory, accumulates traces and initialize model.

        Parameters
        ----------
        blocks : np.ndarray
            Encrypted data blocks for each trace.
        traces : np.ndarray
            Traces matrix.
        model : int
            Model index.

        """
        _, m = traces.shape
        self.n = 0
        self.m = m
        self.hyp = np.empty((COUNT_HYP, COUNT_CLS), dtype=np.uint8)
        self.lens = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_CLS), dtype=np.int)
        self.sums = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_CLS, m))
        self.sums2 = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_CLS, m))
        self.sum = np.zeros(m)
        self.sum2 = np.zeros(m)

        self.accumulate(blocks, traces).init_model(model)

    def accumulate(self, blocks, traces):
        """Sorts traces by class and compute means and deviation.

        Parameters
        ----------
        blocks : np.ndarray
            Encrypted data blocks for each trace.
        traces : np.ndarray
            Traces matrix.

        Returns
        -------
        sca-automation.lib.cpa.Handler
            Reference to self.

        """
        n, _ = traces.shape
        self.n += n
        for i, j, (block, trace) in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN), zip(blocks, traces)):
            k = block[i, j]
            self.lens[i, j, k] += 1
            self.sums[i, j, k] += trace
            self.sums2[i, j, k] += np.square(trace)

        self.sum = np.sum(traces, axis=0)
        self.sum2 = np.sum(traces * traces, axis=0)
        return self

    def init_model(self, model):
        """Initializes power consumption model.

        Parameters
        ----------
        model : int
            Model index.

        Returns
        -------
        sca-automation.lib.cpa.Handler
            Reference to self.

        """
        for h, k in product(range(COUNT_HYP), range(COUNT_CLS)):
            if model == Models.SBOX:
                self.hyp[h, k] = bin(aes.S_BOX[k ^ h] ^ k).count("1")
            elif model == Models.INV_SBOX:
                self.hyp[h, k] = bin(aes.INV_S_BOX[k ^ h] ^ k).count("1")
        return self

    def correlations(self):
        """Computes Pearson's correlation coefficient on current data.

        Returns
        -------
        np.ndarray
            Temporal correlation per block position and hypothesis.

        """
        n, m = self.n, self.m
        ret = np.empty((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_HYP, m))
        mean = self.sum / n
        dev = self.sum2 / n
        dev -= np.square(mean)
        dev = np.sqrt(dev)

        for i, j in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN)):
            mean_ij = np.nan_to_num(self.sums[i, j] / self.lens[i, j].reshape((COUNT_HYP, 1)))
            for h in range(COUNT_HYP):
                y = np.array(self.hyp[h] * self.lens[i, j], dtype=np.float)
                y_mean = np.sum(y) / n
                y_dev = np.sqrt(np.sum(self.hyp[h] * y) / n - y_mean * y_mean)
                xy = np.sum(y.reshape((COUNT_HYP, 1)) * mean_ij, axis=0) / n
                ret[i, j, h] = ((xy - mean * y_mean) / dev) / y_dev
                ret[i, j, h] = np.nan_to_num(ret[i, j, h])

        return ret

    @classmethod
    def guess_stats(cls, cor, key):
        """Computes the best guess key from correlation data.

        Parameters
        ----------
        cor : np.ndarray
            Temporal correlation per block position and hypothesis.
        key : np.ndarray

        Returns
        -------
        guess : np.ndarray
            Guessed key block.
        maxs : np.ndarray
            Maximums of temporal correlation per hypothesis.
        exact : np.ndarray
            ``True`` if the guess is exact for each byte position.

        See Also
        --------
        correlations : Compute temporal correlation.

        """
        maxs = np.max(cor, axis=3)
        guess = np.argmax(maxs, axis=2)
        exact = guess == key
        return guess, maxs, exact

    def guess_envelope(self, cor):
        """Computes the envelope of correlation.

        The envelope consists on two curves representing
        respectively the max and min of temporal correlation
        at each instant.

        This feature is mainly useful to plot
        temporal correlations curve.

        Parameters
        ----------
        cor : np.ndarray
            Temporal correlation per block position and hypothesis.

        Returns
        -------
        cor_max : np.ndarray
            Maximum correlation at each instant.
        cor_min : np.ndarray
            Minimum correlation at each instant.

        See Also
        --------
        correlations : Compute temporal correlation.

        """
        _, _, _, m = self.sums.shape
        cor_max = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, m))
        cor_min = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, m))
        for i, j, t in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN), range(m)):
            cor_max[i, j, t] = np.max(cor[i, j, :, t])
            cor_min[i, j, t] = np.min(cor[i, j, :, t])

        return cor_max, cor_min
