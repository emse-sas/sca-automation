import numpy as np
from lib import aes, cpa
from ui.actions import timed


@timed("updating handler", "update successful!")
def handler(channel, traces, model, current=None):
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
    if current.samples:
        current.accumulate(blocks, traces)
    else:
        current.clear(traces.shape[1]).accumulate(blocks, traces)
    return key


def trace(current, traces):
    if current is None:
        current = np.zeros((traces.shape[1]))
    current += np.sum(traces, axis=0)
    return current
