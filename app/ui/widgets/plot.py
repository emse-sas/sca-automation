from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)


class DoubleGraphFrame(LabelFrame):
    def __init__(self, master, text):
        super().__init__(master, text=text)

        gs_kw = dict(left=0.2, hspace=0.2)
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, gridspec_kw=gs_kw)
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP)


class PlotFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Plots")
        self.acquisition = DoubleGraphFrame(self, "Acquisition")
        self.acquisition.pack(side=TOP, expand=1, fill=Y)
        self.correlation = DoubleGraphFrame(self, "Correlation")
        self.correlation.pack(side=TOP, expand=1, fill=Y)
