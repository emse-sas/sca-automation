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
    means: np.ndarray
        Average trace per class.
    devs: np.ndarray
        Standard deviation trace per class.
    mean: np.ndarray
        Average trace for all classes.
    dev: np.ndarray
        Standard deviation trace for all classes.

    """

    def __init__(self, blocks, key, traces, model=Models.SBOX):
        """Allocates memory, accumulates traces and initialize model.

        Parameters
        ----------
        blocks : np.ndarray
            Encrypted data blocks for each trace.
        key : np.ndarray
            Key data block for all the traces.
        traces : np.ndarray
            Traces matrix.
        model : int
            Model index.

        """
        n, m = traces.shape
        self.blocks = blocks
        self.key = key
        self.hyp = np.empty((COUNT_HYP, COUNT_CLS), dtype=np.uint8)
        self.lens = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_CLS), dtype=np.int)
        self.means = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_CLS, m))
        self.devs = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_CLS, m))
        self.mean = np.zeros(m)
        self.dev = np.zeros(m)

        self.accumulate(traces).init_model(model)

    def accumulate(self, traces):
        """Sorts traces by class and compute means and deviation.

        Parameters
        ----------
        traces : np.ndarray
            Traces matrix.

        Returns
        -------
        sca-automation.lib.cpa.Handler
            Reference to self.

        """
        n, m = traces.shape
        bt = zip(self.blocks, traces)
        for i, j, (block, trace) in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN), bt):
            k = block[i, j]
            self.lens[i, j, k] += 1
            self.means[i, j, k] += trace
            self.devs[i, j, k] += np.square(trace)

        for i, j, k in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN), range(COUNT_CLS)):
            if self.lens[i, j, k] == 0:
                continue
            self.means[i, j, k] /= self.lens[i, j, k]
            self.devs[i, j, k] /= self.lens[i, j, k]
            self.devs[i, j, k] -= np.square(self.means[i, j, k])
            self.devs[i, j, k] = np.sqrt(self.devs[i, j, k])

        self.mean = np.sum(traces, axis=0) / n
        self.dev = np.sum(traces * traces, axis=0) / n
        self.dev -= np.square(self.mean)
        self.dev = np.sqrt(self.dev)
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
        n = len(self.blocks)
        _, _, _, m = self.means.shape
        ret = np.empty((aes.BLOCK_LEN, aes.BLOCK_LEN, COUNT_HYP, m))
        for i, j, h in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN), range(COUNT_HYP)):
            y = np.array(self.hyp[h] * self.lens[i, j], dtype=np.float)
            y_mean = np.sum(y) / n
            y_std = np.sqrt(np.sum(self.hyp[h] * y) / n - y_mean * y_mean)
            xy = np.sum(y.reshape((COUNT_HYP, 1)) * self.means[i, j], axis=0) / n
            ret[i, j, h] = ((xy - self.mean * y_mean) / self.dev) / y_std
            ret[i, j, h] = np.nan_to_num(ret[i, j, h])

        return ret

    def guess_stats(self, cor):
        """Computes the best guess key from correlation data.

        Parameters
        ----------
        cor : np.ndarray
            Temporal correlation per block position and hypothesis.

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
        exact = guess == self.key
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
        _, _, _, m = self.means.shape
        cor_max = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, m))
        cor_min = np.zeros((aes.BLOCK_LEN, aes.BLOCK_LEN, m))
        for i, j, t in product(range(aes.BLOCK_LEN), range(aes.BLOCK_LEN), range(m)):
            cor_max[i, j, t] = np.max(cor[i, j, :, t])
            cor_min[i, j, t] = np.min(cor[i, j, :, t])

        return cor_max, cor_min
