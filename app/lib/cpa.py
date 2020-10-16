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

from lib import aes, traces as tr

COUNT_HYP = 256  # Count of key hypothesis for one byte
COUNT_CLS = 256  # Traces with the same byte value in a given position
BLOCK_SIZE = aes.BLOCK_SIZE


class Models:
    """CPA power consumption models.

    """

    SBOX = 0
    INV_SBOX = 1


class Statistics:
    def __init__(self, handler=None):
        self.corr = None
        self.corr_min = None
        self.corr_max = None

        self.key = None
        self.corr_key = []
        self.corr_guess = []
        self.guesses = []
        self.exacts = []
        self.ranks = []
        self.maxs = []
        self.iterations = []
        self.divs = []

        if handler and handler.iterations > 0:
            self.update(handler)

    def update(self, handler):
        self.corr = handler.correlations()
        self.key = handler.key
        guess, mx, exact, rank, corr_key, corr_guess = Statistics.guess_stats(self.corr, handler.key)
        self.corr_max, self.corr_min = Statistics.guess_envelope(self.corr, guess)
        self.corr_key.append(corr_key)
        self.corr_guess.append(corr_guess)
        self.guesses.append(guess)
        self.exacts.append(exact)
        self.ranks.append(rank)
        self.maxs.append(mx)
        self.iterations.append(handler.iterations)
        self.divs.append(self.div_idxs())

    def clear(self):
        self.corr = None
        self.corr_min = None
        self.corr_max = None
        self.corr_key.clear()
        self.corr_guess.clear()
        self.guesses.clear()
        self.exacts.clear()
        self.ranks.clear()
        self.maxs.clear()
        self.iterations.clear()

    def __repr__(self):
        return f"Statistics({self.corr})"

    def __str__(self):
        ret = f"{'Byte':<8}{'Exact':<8}{'Key':<8}{'(%)':<8}{'Guess':<8}{'(%)':<8}{'Rank':<8}{'Divergence':<8}\n"
        for b in range(BLOCK_SIZE):
            ret += f"{b:<8}" \
                   f"{bool(self.exacts[-1][b]):<8}" \
                   f"{self.key[b]:<8x}{100 * self.corr_key[-1][b]:<5.2f}{'%':<3}" \
                   f"{self.guesses[-1][b]:<8x}{100 * self.corr_guess[-1][b]:<5.2f}{'%':<3}" \
                   f"{self.ranks[-1][b]:<8}" \
                   f"{self.divs[-1][b]:<8}\n"

        return ret

    @classmethod
    def graph(cls, data):
        data = np.array(data)
        n = len(data.shape)
        r = tuple(range(n))
        return np.moveaxis(data, r, tuple([r[-1]] + list(r[:-1])))

    def div_idxs(self, n=0.2):
        div = np.full((BLOCK_SIZE,), fill_value=-1)
        for b in range(BLOCK_SIZE):
            if self.key[b] != self.guesses[-1][b]:
                continue
            for chunk, mx in enumerate(self.maxs):
                mx_second = mx[b, np.argsort(mx[b])[-2]]
                mx_key = mx[b, self.key[b]]
                if (mx_key - mx_second) / mx_key > n:
                    div[b] = self.iterations[chunk]
                    break
        return div

    @classmethod
    def guess_stats(cls, cor, key):
        """Computes the best guess key from correlation data.

        Parameters
        ----------
        cor : np.ndarray
            Temporal correlation per block byte and hypothesis.
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
        best = np.amax(cor, axis=2)
        guess = np.argmax(best, axis=1)
        rank = COUNT_HYP - np.argsort(np.argsort(best, axis=1), axis=1)
        rank = np.array([rank[b, key[b]] for b in range(BLOCK_SIZE)])
        corr_guess = np.array([best[b, guess[b]] for b in range(BLOCK_SIZE)])
        corr_key = np.array([best[b, key[b]] for b in range(BLOCK_SIZE)])
        exact = guess == key
        return guess, best, exact, rank, corr_key, corr_guess

    @classmethod
    def guess_envelope(cls, cor, guess):
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
        guess : np.ndarray
            Guessed block matrix.
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
        env = np.moveaxis(cor.copy(), (0, 1, 2), (0, 2, 1))
        for b in range(BLOCK_SIZE):
            env[b, :, guess[b]] -= env[b, :, guess[b]]

        return np.max(env, axis=2), np.min(env, axis=2)


class Handler:
    """CPA correlation handler interface.

    Attributes
    ----------
    blocks: np.ndarray
        Encrypted data blocks for each trace.
    key: np.ndarray
        Key data block for all the traces.
    hypothesis: np.ndarray
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

    def __init__(self, model, channel=None, traces=None, samples=None):
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
        self.model = model
        self.blocks = None
        self.key = None
        self.iterations = 0
        self.samples = None
        self.hypothesis = None
        self.lens = None

        self.sums = None
        self.sums2 = None
        self.sum = None
        self.sum2 = None

        if traces is not None and channel is not None:
            samples = samples or traces.shape[1]
            self.clear(samples).set_model(model).set_key(channel).set_blocks(channel).accumulate(traces)
        else:
            self.clear(samples or 0).set_model(model)

    def clear(self, samples=0):
        self.iterations = 0
        self.samples = samples
        self.hypothesis = np.zeros((COUNT_HYP, COUNT_CLS), dtype=np.uint8)
        self.lens = np.zeros((BLOCK_SIZE, COUNT_CLS), dtype=np.int)
        self.sums = np.zeros((BLOCK_SIZE, COUNT_CLS, samples), dtype=np.float)
        self.sums2 = np.zeros((BLOCK_SIZE, COUNT_CLS, samples), dtype=np.float)
        self.sum = np.zeros(samples, dtype=np.float)
        self.sum2 = np.zeros(samples, dtype=np.float)
        return self

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
        for b, (block, trace) in product(range(BLOCK_SIZE), zip(self.blocks, traces)):
            k = block[b]
            self.lens[b, k] += 1
            self.sums[b, k] += trace
            self.sums2[b, k] += np.square(trace)
        self.iterations += traces.shape[0]
        self.sum += np.sum(traces, axis=0)
        self.sum2 += np.sum(traces * traces, axis=0)
        return self

    def set_key(self, channel):
        shape = (BLOCK_SIZE,)
        if self.model == Models.SBOX:
            self.key = aes.words_to_block(channel.keys[0]).reshape(shape)
        elif self.model == Models.INV_SBOX:
            self.key = aes.key_expansion(aes.words_to_block(channel.keys[0]))[10].T.reshape(shape)
        else:
            raise ValueError(f"unknown model: {self.model}")
        return self

    def set_blocks(self, channel):
        shape = (BLOCK_SIZE,)
        if self.model == Models.SBOX:
            self.blocks = np.array([aes.words_to_block(block).reshape(shape) for block in channel.plains])
        elif self.model == Models.INV_SBOX:
            self.blocks = ([aes.words_to_block(block).reshape(shape) for block in channel.ciphers])
        else:
            raise ValueError(f"unknown model: {self.model}")
        return self

    def set_model(self, model):
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
        self.model = model
        if model == Models.SBOX:
            for h, k in product(range(COUNT_HYP), range(COUNT_CLS)):
                self.hypothesis[h, k] = bin(aes.S_BOX[k ^ h]).count("1")
        elif model == Models.INV_SBOX:
            for h, k in product(range(COUNT_HYP), range(COUNT_CLS)):
                self.hypothesis[h, k] = bin(aes.INV_S_BOX[k ^ h] ^ k).count("1")
        else:
            raise ValueError(f"unknown model: {self.model}")
        return self

    def correlations(self):
        """Computes Pearson's correlation coefficient on current data.

        Returns
        -------
        np.ndarray
            Temporal correlation per block position and hypothesis.

        """
        n = self.iterations
        ret = np.empty((BLOCK_SIZE, COUNT_HYP, self.samples))
        mean = self.sum / n
        dev = self.sum2 / n
        dev -= np.square(mean)
        dev = np.sqrt(dev)

        for b in range(BLOCK_SIZE):
            mean_ij = np.nan_to_num(self.sums[b] / self.lens[b].reshape((COUNT_CLS, 1)))
            for h in range(COUNT_HYP):
                y = np.array(self.hypothesis[h] * self.lens[b], dtype=np.float)
                y_mean = np.sum(y) / n
                y_dev = np.sqrt(np.sum(self.hypothesis[h] * y) / n - y_mean * y_mean)
                xy = np.sum(y.reshape((COUNT_HYP, 1)) * mean_ij, axis=0) / n
                ret[b, h] = ((xy - mean * y_mean) / dev) / y_dev
                ret[b, h] = np.nan_to_num(ret[b, h])

        return ret
