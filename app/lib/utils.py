"""General utility functions.

"""
import functools
import os
import time
import matplotlib.pyplot as plt
from datetime import timedelta


def decode_hamming(c, offset=0):
    """Decodes ASCII encoded hamming weight.
    
    Parameters
    ----------
    c : str
        Char representing encoded hamming weight.
    offset : int, optional
        Additional offset value used to encode weight.

    Returns
    -------
    int
        Decoded hamming weight.
    """
    return int(c) - ord("P") + offset


def check_hex(w):
    """Checks if a correct hexadecimal bytes string is given.

    Parameters
    ----------
    w : str
        Hexadecimal 32-bit word string.

    Returns
    -------
    0 padded hexadecimal string.

    Raises
    ------
    ValueError
        Unable to parse integer from given string.

    """
    return f"{int(w, 16):08x}"


def format_sizeof(num, suffix="B"):
    """Converts and format a number to a file size unit.

    Parameters
    ----------
    num : int
        Number to format.
    suffix : str
        Unit suffix.
    Returns
    -------
    str
        Formatted number.
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}{'Yi'}{suffix}"


def create_subdir(names, path=None):
    if path:
        try_create_dir(path)
    for name in names:
        try_create_dir(os.path.join(path, name))


def remove_subdir_files(path):
    """Removes files in the sub directories at given directory.

    Files in the given directory are not deleted.

    Parameters
    ----------
    path : str
        Path where to remove sub directories.

    """
    for dir_path, _, filenames in os.walk(path):
        for filename in filenames:
            os.remove(os.path.join(dir_path, filename))


def try_create_dir(path):
    """Creates directory or print a message
.
    Parameters
    ----------
    path : str
        Path to the directory to create.

    """
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"{path} already exists")
        pass


def operation_decorator(title, message):
    """Executes a function and prints messages and duration.

    Parameters
    ----------
    title : str
        Starting message.

    message : str
        End message on success.
    Returns
    -------
        function
            Decorated method.

    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            print(f"\n*** {title} ***")
            t_start = time.perf_counter()
            result = function(*args, **kwargs)
            t_end = time.perf_counter()
            print(f"{message}\nelapsed: {str(timedelta(seconds=t_end - t_start))}")
            return result

        return wrapper

    return decorator


def plot_decorator(title, xlabel, ylabel, filepath):
    """Performs plot methods and formatting.

    Parameters
    ----------
    title : str
        Title of the plot.
    xlabel : str
        X-axis legend.
    ylabel : str
        Y-axis legend.
    filepath : str
        Path to save plot image.

    Returns
    -------
    function
        Decorated method.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(filepath)
            plt.show()
            plt.close()
            return result

        return wrapper

    return decorator
