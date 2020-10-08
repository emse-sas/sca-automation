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
import sys
from datetime import datetime

import numpy as np
import ui.actions
import ui.update
import ui.plot
from lib.data import Request, Parser
from lib.cpa import Handler
from lib import traces as tr
from scipy import fft, signal
from warnings import warn
from threading import Thread
from multiprocessing import Process


class Threads:
    parse = None
    acquire = None
    correlate = None


@ui.actions.timed("attack.py", "\nexiting...")
def main(args):
    f_c = 13e6
    order = 4
    w = f_c / (200e6 / 2)
    b, a, *_ = signal.butter(order, w, btype="highpass", output="ba")

    @ui.actions.timed("start acquisition")
    def prepare(chunk=None):
        if chunk is not None:
            print(f"{'chunk':<16}{chunk + 1}/{request.chunks}")
            print(f"{'requested':<16}{(chunk + 1) * request.iterations}/{request.iterations * request.chunks}")
        print(f"{'started':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")

    @ui.actions.timed("start processing", "\nprocessing successful!")
    def process(x, chunk):
        if Threads.parse is not None:
            Threads.parse.join()
        Threads.parse = Thread(target=parse, args=[x, chunk])
        Threads.parse.start()

    def parse(x, chunk):
        parser = Parser(x, direction=request.direction, verbose=request.verbose)
        parsed = len(parser.channel)
        try:
            ui.save(request, x, parser.leak, parser.channel, parser.meta, parser.noise, chunk=chunk, path=savepath)
        except OSError as e:
            print(f"Fatal error : {e}\nexiting...")
            sys.exit(1)

        print(f"{'chunk':<16}{chunk + 1}/{request.chunks}")
        print(f"{'size':<16}{ui.sizeof(len(x or []))}")
        print(f"{'parsed':<16}{parsed}/{request.iterations}")
        print(f"{'total':<16}{parser.meta.iterations}/{(request.chunks or 1) * request.iterations}")

        if not parsed:
            warn("no traces parsed!\nskipping...")
            return

        trace = ui.update.Current.trace
        m = len(trace if trace is not None else [])
        traces = np.array(tr.adjust(parser.leak.traces, m))

        Thread.acquire = Thread(target=acquire, args=(parser, traces))
        Thread.correlate = Thread(target=correlate, args=(parser, traces.copy()))

        Thread.acquire.start()
        Thread.correlate.start()

        Thread.acquire.join()
        Thread.correlate.join()

    def acquire(parser, traces):
        mean = ui.update.trace(traces) / parser.meta.iterations
        spectrum = np.absolute(fft.fft(mean - np.mean(mean)))
        ui.plot.acquisition(traces, mean, spectrum, parser.meta, request, path=savepath)

    def correlate(parser, traces):
        for trace in traces:
            trace[:] = signal.filtfilt(b, a, trace)
        key = ui.update.handler(parser.channel, traces, model=args.model)
        cor = ui.update.Current.handler.correlations()
        ui.plot.correlations(cor, key, request, maxs, ui.update.Current.handler, path=loadpath)

    maxs = []
    ui.update.Current.handler = Handler(model=args.model)
    request = Request(args)
    savepath, loadpath = ui.init(request, args.path)
    print(request)
    print(f"{'load path':<16}{os.path.abspath(loadpath)}")
    print(f"{'save path':<16}{os.path.abspath(savepath)}")
    ui.actions.acquire(request, process, prepare=prepare, path=loadpath)


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
argp.add_argument("-s", "--source",
                  choices=[Request.Sources.FILE, Request.Sources.SERIAL],
                  default=Request.Sources.FILE,
                  help="Acquisition source.")
argp.add_argument("-p", "--plot", type=int, default=16,
                  help="Count of raw traces to plot.")
argp.add_argument("--start", type=int,
                  help="Start time sample index of each trace.")
argp.add_argument("--end", type=int,
                  help="End time sample index of each trace.")
argp.add_argument("-v", "--verbose", action="store_true",
                  help="End time sample index of each trace.")
argp.add_argument("--model", type=int,
                  help="Leakage model.")

if __name__ == "__main__":
    main(argp.parse_args())
