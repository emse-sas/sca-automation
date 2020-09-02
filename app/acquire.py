"""Export and plot data from SoC.

The data retrieved from SoC consists on side-channel leakage
and encryption data in order to later perform correlation.

The side-channel leakage is plot to evaluate the quality of
the acquisition.
The data is exported in 3 separated CSV files.

Examples
--------
.. code-block:: shell

    $ python acquire.py 256 COM5
    $ python acquire.py -m sw 1024 cmd

In the above example, the first line will launch the acquisition
of 256 hardware traces and read the serial port ``COM5``.

The second line will read the file ``cmd_hw_1024.log`` located in the
directory containing the 1024 software traces.

In the last case, the file must be a valid binary log file previously
acquired from the SoC via serial port.

"""

import argparse

import ui
from lib.utils import operation_decorator
from lib.data import Request


@operation_decorator("acquire.py", "\nexiting...")
def main(args):
    def callback(x, chunk=None):
        if chunk is not None:
            print(f"\nchunk: {chunk + 1}/{args.chunks}")
        parser = ui.parse(x, request)
        ui.save(request, parser.leak, parser.channel, parser.meta)
        ui.plot_acq(parser.leak, parser.meta, request)

    request = Request(args)
    if hasattr(args, "chunks"):
        ui.acquire_chunks(request, callback)
    else:
        callback(ui.acquire(request))


argp = argparse.ArgumentParser(
    description="Acquire data from SoC and export it.")
argp.add_argument("iterations", type=int,
                  help="Requested count of traces.")
argp.add_argument("name", type=str,
                  help="Acquisition source name.")
argp.add_argument("-m", "--mode",
                  choices=[Request.Modes.HARDWARE, Request.Modes.SOFTWARE],
                  default=Request.Modes.HARDWARE,
                  help="Encryption mode.")
argp.add_argument("-d", "--direction",
                  choices=[Request.Directions.ENCRYPT, Request.Directions.DECRYPT],
                  default=Request.Directions.ENCRYPT,
                  help="Encryption direction.")
argp.add_argument("-s", "--source",
                  choices=[Request.Sources.FILE, Request.Sources.SERIAL],
                  default=Request.Sources.FILE,
                  help="Acquisition source.")
argp.add_argument("-p", "--plot", type=int, default=16,
                  help="Count of raw traces to plot.")
argp.add_argument("-c", "--chunks", type=int, default=None,
                  help="Count of chunks to acquire.")

if __name__ == "__main__":
    main(argp.parse_args())
