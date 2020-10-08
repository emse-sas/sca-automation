"""Import acquisition data and perform attack.

The data are imported from CSV files produced by ``acquire.py``
in order to avoid parsing which is computationally expensive.

The temporal correlations are plot and the key guess is displayed
to validate the attack.

Examples
--------
.. code-block:: shell

    $ python correlate.py 256
    $ python correlate.py -m sw 1024

In the above example, the first line will launch
the attack on the CSV files representing the SoC data retrieved
from the acquisition of 256 hardware traces.

The second example do the same except it will launch the attack
on 1024 software traces.

"""
import argparse
import os
from datetime import datetime

import numpy as np
import ui.actions
import ui.update
import ui.plot
from lib.data import Request
from lib.cpa import Handler
from lib import traces as tr
from scipy import signal


@ui.actions.timed("correlate.py", "\nexiting...")
def main(args):
    f_c = 13e6
    order = 4
    w = f_c / (200e6 / 2)
    b, a, *_ = signal.butter(order, w, btype="highpass", output="ba")

    @ui.actions.timed("start acquisition")
    def prepare(chunk=None):
        if chunk is not None:
            print(f"{'chunk':<16}{chunk + 1}/{args.chunks}")
            print(f"{'requested':<16}{(chunk + 1) * request.iterations}/{request.iterations * request.chunks}")
        print(f"{'started':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")

    @ui.actions.timed("start processing", "\nprocessing successful!")
    def process(channel, leak, meta, _):
        print(f"{'started':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")
        m = handler.samples if handler.samples else None
        traces = np.array(tr.adjust(leak.traces, m), dtype=np.float)
        for trace in traces:
            trace[:] = signal.filtfilt(b, a, trace)
        key = ui.update.handler(channel, traces, model=args.model, current=handler)
        cor = handler.correlations()
        ui.plot.correlations(cor, key, request, maxs, handler, path=loadpath)

    maxs = []
    handler = Handler(model=args.model)
    request = Request(args)
    _, loadpath = ui.init(request, args.path)
    print(request)
    print(f"{'load path':<16}{os.path.abspath(loadpath)}")
    ui.actions.load(request, process, prepare, path=loadpath)


np.set_printoptions(formatter={"int": hex})
argp = argparse.ArgumentParser(
    description="Load acquisition data and perform a side-channel attack.")
argp.add_argument("iterations", type=int,
                  help="Requested count of traces.")
argp.add_argument("name", type=str,
                  help="Acquisition source name.")
argp.add_argument("-m", "--mode",
                  choices=[Request.Modes.HARDWARE, Request.Modes.TINY, Request.Modes.SSL],
                  default=Request.Modes.HARDWARE,
                  help="Encryption mode.")
argp.add_argument("--chunks", type=int, default=None,
                  help="Count of chunks to acquire.")
argp.add_argument("--path", type=str, default=ui.actions.DEFAULT_DATA_PATH,
                  help="Path where to save files.")
argp.add_argument("--model", type=int,
                  help="Leakage model.")

if __name__ == "__main__":
    main(argp.parse_args())
