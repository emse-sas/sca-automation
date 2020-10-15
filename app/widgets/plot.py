from tkinter import *

import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from lib.cpa import COUNT_HYP, Statistics
from lib.aes import BLOCK_LEN


def raw(ax, traces, limit=16, chunk=None):
    chunk = (chunk or 0) + 1
    ax.set(xlabel="Time Samples", ylabel="Hamming Weights")
    return [ax.plot(trace, label=f"iteration {d * chunk}") for d, trace in enumerate(traces[:limit])]


def avg(ax, trace):
    ax.set(xlabel="Time Samples", ylabel="Hamming Weights")
    return ax.plot(trace, color="grey")


def fft(ax, freq, spectrum, f):
    ax.set(xlabel="Frequency (MHz)", ylabel="Hamming Weight")
    return ax.plot(freq[f], spectrum[f], color="red")


def iterations(ax, scale, stats, idx):
    maxs = Statistics.graph(stats.maxs)
    i, j = idx
    ax.set(xlabel="Traces acquired", ylabel="Pearson Correlation")
    plot_key = None
    plot_guess = None
    plots = []
    for h in range(COUNT_HYP):
        if h == stats.key[i, j] and h == stats.guesses[-1][i, j]:
            plot_key, = plot_guess, = ax.plot(scale, maxs[i, j, h], color="r", zorder=10)
        elif h == stats.key[i, j]:
            plot_key, = ax.plot(scale, maxs[i, j, h], color="b", zorder=10)
        elif h == stats.guesses[-1][i, j]:
            plot_guess, = ax.plot(scale, maxs[i, j, h], color="c", zorder=10)
        else:
            plots.append(ax.plot(scale, maxs[i, j, h], color="grey"))
    return plot_key, plot_guess, plots


def temporal(ax, stats, idx):
    i, j = idx
    corr_guess = stats.corr[i, j, stats.guesses[-1][i, j]]
    corr_key = stats.corr[i, j, stats.key[i, j]]
    ax.set(xlabel="Time Samples", ylabel="Pearson Correlation")
    ax.fill_between(range(stats.corr.shape[3]), stats.corr_max[i, j], stats.corr_min[i, j], color="grey")
    if stats.exacts[-1][i, j]:
        plot_key, = plot_guess, = ax.plot(corr_guess, color="r")
    else:
        plot_guess, = ax.plot(corr_guess, color="c")
        plot_key, = ax.plot(corr_key, color="b")
    return plot_key, plot_guess


class PlotFrame(LabelFrame):
    def __init__(self, master, scale=None):
        super().__init__(master, text="Plots")
        self.scale = scale or []
        self.gs = gridspec.GridSpec(2, 1, hspace=0.5)
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(self.gs[0])
        self.ax2 = self.fig.add_subplot(self.gs[1])
        self.plot1 = None
        self.plot2 = None
        self.annotation = None
        self.gs.tight_layout(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP)

    def clear(self):
        self.scale = []
        self.ax1.clear()
        self.ax2.clear()
        self.fig.clear()
        self.gs.tight_layout(self.fig)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.canvas.figure = self.fig
        self.canvas.draw()

    def draw_stats(self, data):
        mean, spectrum, freq = data
        for legend in self.fig.legends:
            legend.remove()
        self.ax1.clear()
        self.ax2.clear()
        self.plot1, = avg(self.ax1, mean)
        self.plot2, = fft(self.ax2, freq, spectrum, freq)
        self.fig.suptitle("Leakage statistics")
        self.fig.legend((self.plot1, self.plot2),
                        ("Temporal average", "Spectrum average"))
        self.fig.canvas.draw()

    def update_scale(self, handler, request):
        self.scale.append(handler.iterations)
        if not request.chunks:
            return
        self.ax1.set_xlim([self.scale[0], request.iterations * (request.chunks or 1)])

    def draw_corr(self, data, idx):
        i, j = idx
        stats = data
        for legend in self.fig.legends:
            legend.remove()
        self.ax1.clear()
        self.ax2.clear()
        plot_key, plot_guess, _ = iterations(self.ax1, self.scale, stats, idx)
        temporal(self.ax2, stats, idx)
        self.fig.suptitle(f"Correlation byte {i * BLOCK_LEN + j}")
        self.fig.legend((plot_key, plot_guess),
                        (f"key 0x{stats.key[i, j]:02x}", f"guess 0x{stats.guesses[-1][i, j]:02x}"))
        self.fig.canvas.draw()
