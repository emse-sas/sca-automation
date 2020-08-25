import argparse

import core

from lib.utils import operation_decorator


@operation_decorator("clean.py", "\nexiting...")
def main(args):
    if args.log:
        core.remove_logs()
    if args.acq:
        core.remove_acquisition_images()
    if args.cor:
        core.remove_correlation_images()


argp = argparse.ArgumentParser(
    description="Clean the data and the medias produced by the application.")
argp.add_argument("-l", "--log", action="store_true",
                  help="Remove log files.")
argp.add_argument("-a", "--acq", action="store_true",
                  help="Remove acquisition images.")
argp.add_argument("-c", "--cor", action="store_true",
                  help="Remove correlation images.")

if __name__ == "__main__":
    main(argp.parse_args())
