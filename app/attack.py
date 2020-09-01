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

import core
from lib.utils import operation_decorator


@operation_decorator("attack.py", "\nexiting...")
def main(args):
    request = core.Request.from_args(args)
    leak, data, meta = core.import_csv(request)
    traces, _, _ = core.filter_traces(leak)
    handler = core.init_handler(data, traces, model=args.model)
    core.plot_cor(handler, request)


np.set_printoptions(formatter={"int": hex})
argp = argparse.ArgumentParser(
    description="Load acquisition data and perform a side-channel attack.")
argp.add_argument("-m", "--mode", choices=core.MODES, default="hw",
                  help="Encryption mode.")
argp.add_argument("iterations", type=int,
                  help="Requested count of traces.")
argp.add_argument("model", type=int,
                  help="Leakage model.")

if __name__ == "__main__":
    main(argp.parse_args())
