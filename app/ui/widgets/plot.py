from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

import ui
from lib.aes import BLOCK_LEN


class DoublePlotFrame(LabelFrame):
    def __init__(self, master, text):
        super().__init__(master, text=text)

        gs_kw = dict(left=0.2, hspace=0.2)
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=False, gridspec_kw=gs_kw)
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.plot1 = None
        self.plot2 = None
        self.annotation = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP)


class AcquisitionPlotFrame(DoublePlotFrame):
    async def draw(self, mean, spectrum, freq, annotation):
        self.ax1.clear()
        self.ax2.clear()
        self.plot1 = ui.plot.mean(self.ax1, mean)
        self.plot2 = ui.plot.fft(self.ax2, freq, spectrum, freq)
        self.annotation = ui.plot.annotate(self.ax1, annotation)
        self.fig.suptitle("Filtered power consumptions")
        self.fig.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_event()


class CorrelationPlotFrame(DoublePlotFrame):
    def __init__(self, master, text, scale=None):
        super().__init__(master, text)
        self.scale = scale or []

    async def update_scale(self, handler, request):
        self.scale.append(handler.iterations)
        self.ax1.set_xlim([self.scale[0], request.iterations * request.chunks])

    async def draw(self, i, j, key, cor, guess, maxs_graph, exacts, max_env, min_env, annotation):
        b = i * BLOCK_LEN + j
        self.ax1.clear()
        self.ax2.clear()
        ui.plot.iterations(self.ax1, self.scale, guess[i, j], key[i, j], maxs_graph[i, j])
        self.ax2.fill_between(range(cor.shape[3]), max_env[i, j], min_env[i, j], color="grey")
        ui.plot.temporal(self.ax2, cor[i, j, guess[i, j]], cor[i, j, key[i, j]], guess[i, j], key[i, j], exacts[i, j])
        ui.plot.annotate(self.ax1, annotation)
        self.fig.suptitle(f"Correlation byte {b}")
        self.fig.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_event()


class PlotFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Plots")
        self.acquisition = AcquisitionPlotFrame(self, "Acquisition")
        self.acquisition.pack(side=TOP, expand=1, fill=Y)
        self.correlation = CorrelationPlotFrame(self, "Correlation")
        self.correlation.pack(side=TOP, expand=1, fill=Y)
