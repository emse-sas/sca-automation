"""Import acquisition data and perform attack.

The data are imported from CSV files produced by ``acquire.py``
in order to avoid parsing which is computationally expensive.

The temporal correlations are plot and the key guess is displayed
to validate the attack.

Examples
--------
.. code-block:: shell

    $ python attack.py 256
    $ python attack.py -m sw 1024

In the above example, the first line will launch
the attack on the CSV files representing the SoC data retrieved
from the acquisition of 256 hardware traces.

The second example do the same except it will launch the attack
on 1024 software traces.

"""

import argparse

import numpy as np

import ui
from lib.utils import operation_decorator
from lib.data import Request


@operation_decorator("attack.py", "\nexiting...")
def main(args):
    @operation_decorator("process chunk", "\nsuccess...")
    def callback(channel, leak, meta, chunk=None):
        h = handler
        if chunk is not None:
            print(f"\nchunk: {chunk + 1}/{args.chunks}")
        print(f"traces imported: {len(channel)}/{request.iterations}")
        traces, _, _ = ui.filter_traces(leak)
        h, key = ui.update_handler(channel, traces, model=args.model, handler=h)
        ui.plot_cor(h, key, request, meta=meta)

    handler = None
    request = Request(args)
    ui.load(request, callback)


np.set_printoptions(formatter={"int": hex})
argp = argparse.ArgumentParser(
    description="Load acquisition data and perform a side-channel attack.")
argp.add_argument("iterations", type=int,
                  help="Requested count of traces.")
argp.add_argument("model", type=int,
                  help="Leakage model.")
argp.add_argument("-m", "--mode",
                  choices=[Request.Modes.HARDWARE, Request.Modes.SOFTWARE],
                  default=Request.Modes.HARDWARE,
                  help="Encryption mode.")
argp.add_argument("-c", "--chunks", type=int, default=None,
                  help="Count of chunks to acquire.")

if __name__ == "__main__":
    main(argp.parse_args())
