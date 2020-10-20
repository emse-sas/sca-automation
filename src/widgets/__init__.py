from tkinter import *

from widgets.config import ConfigFrame
from widgets.log import LogFrame
from widgets.plot import PlotFrame


def sizeof(num, suffix="B"):
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
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} {'Yi'}{suffix}"


class MainFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.panel = Frame(master)
        self.panel.pack(side=LEFT, expand=1, fill=Y)
        self.config = ConfigFrame(self.panel)
        self.config.pack(side=TOP, expand=1, fill=BOTH)
        self.log = LogFrame(self.panel)
        self.log.pack(side=TOP)
        self.plot = PlotFrame(master)
        self.plot.pack(side=RIGHT, expand=1, fill=Y)
        self.buttons = Frame(self.panel)

        self.buttons.pack(side=TOP, expand=1, fill=BOTH)
        self.button_launch = Button(self.buttons, text="Launch", command=self.launch)
        self.button_launch.pack(side=LEFT)
        self.button_stop = Button(self.buttons, text="Stop", command=self.stop)
        self.button_stop.pack(side=LEFT)

        self.clicked_launch = False
        self.clicked_stop = False

    def launch(self):
        self.clicked_launch = True

    def stop(self):
        self.clicked_stop = True

    def lock_launch(self):
        self.button_launch['state'] = DISABLED

    def unlock_launch(self):
        self.button_launch['state'] = NORMAL
