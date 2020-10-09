from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

import numpy as np

from lib.cpa import Models
from lib.data import Request
from ui.widgets.config import GeneralFrame, PerfsFrame, FilesFrame, ConfigFrame
from ui.widgets.log import LogFrame
from ui.widgets.plot import PlotFrame

plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams["figure.titlesize"] = "x-large"


class Main:
    def __init__(self, master):
        self.frame_text = Frame(master)
        self.frame_text.pack(side=LEFT, expand=1, fill=Y)
        self.frame_config = ConfigFrame(self.frame_text)
        self.frame_config.pack(side=TOP, expand=1, fill=BOTH)
        self.frame_log = LogFrame(self.frame_text)
        self.frame_log.pack(side=TOP)
        self.frame_plot = PlotFrame(master)
        self.frame_plot.pack(side=RIGHT, expand=1, fill=Y)

        self.frame_buttons = Frame(self.frame_text)
        self.frame_buttons.pack(side=TOP, expand=1, fill=BOTH)

        self.button_launch = Button(self.frame_buttons, text="Launch", command=self.launch)
        self.button_launch.pack(side=LEFT)
        self.button_stop = Button(self.frame_buttons, text="Stop", command=self.launch)
        self.button_stop.pack(side=LEFT)

    def launch(self):
        pass

    def stop(self):
        pass


root = Tk()
root.title("Side-channel attacks demo")
w = Main(root)

while True:
    root.update()
