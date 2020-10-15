from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

import ui
from lib.aes import BLOCK_LEN


class DoublePlotFrame(LabelFrame):
    def __init__(self, master, text):
        super().__init__(master, text=text)
        self.fig = Figure(figsize=(16, 4))
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)

        self.plot1 = None
        self.plot2 = None
        self.annotation = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP)

    def clear(self):
        self.ax1.clear()
        self.ax2.clear()
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.canvas.figure = self.fig
        self.canvas.draw()


class AcquisitionPlotFrame(DoublePlotFrame):
    def draw(self, mean, spectrum, freq, annotation):
        for legend in self.fig.legends:
            legend.remove()
        self.ax1.clear()
        self.ax2.clear()
        self.plot1, = ui.plot.mean(self.ax1, mean)
        self.plot2, = ui.plot.fft(self.ax2, freq, spectrum, freq)
        self.annotation = ui.plot.annotate(self.ax1, annotation)
        self.fig.suptitle("Filtered power consumptions")
        self.fig.legend((self.plot1, self.plot2), ("Temporal average", "Spectrum average"))
        self.fig.canvas.draw()


class CorrelationPlotFrame(DoublePlotFrame):
    def __init__(self, master, text, scale=None):
        super().__init__(master, text)
        self.scale = scale or []

    def update_scale(self, handler, request):
        self.scale.append(handler.iterations)
        if not request.chunks:
            return
        self.ax1.set_xlim([self.scale[0], request.iterations * (request.chunks or 1)])

    def clear(self):
        self.scale = []
        super().clear()

    def draw(self, i, j, key, cor, guess, maxs_graph, exacts, max_env, min_env, annotation):
        b = i * BLOCK_LEN + j
        for legend in self.fig.legends:
            legend.remove()
        self.ax1.clear()
        self.ax2.clear()
        plot_key, plot_guess = ui.plot.iterations(self.ax1, self.scale, guess[i, j], key[i, j], maxs_graph[i, j])
        self.ax2.fill_between(range(cor.shape[3]), max_env[i, j], min_env[i, j], color="grey")
        ui.plot.temporal(self.ax2, cor[i, j, guess[i, j]], cor[i, j, key[i, j]], guess[i, j], key[i, j], exacts[i, j])
        ui.plot.annotate(self.ax1, annotation)
        self.fig.legend((plot_key, plot_guess), (f"key 0x{key[i, j]:02x}", f"guess 0x{guess[i, j]:02x}"))
        self.fig.suptitle(f"Correlation byte {b}")
        self.fig.canvas.draw()


class PlotFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Plots")
        self.acquisition = AcquisitionPlotFrame(self, "Acquisition")
        self.acquisition.pack(side=TOP, expand=1, fill=Y)
        self.correlation = CorrelationPlotFrame(self, "Correlation")
        self.correlation.pack(side=TOP, expand=1, fill=Y)

    def clear(self):
        self.acquisition.clear()
        self.correlation.clear()