import asyncio
import multiprocessing as mp
import os
import threading
from datetime import datetime
from tkinter import *
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fft

import ui
from lib import aes, cpa
from lib.aes import BLOCK_LEN
from lib.cpa import Handler
from lib.data import Request, Parser
import lib.traces as tr
from ui.widgets.config import GeneralFrame, PerfsFrame, FilesFrame, ConfigFrame
from ui.widgets.log import LogFrame
from ui.widgets.plot import PlotFrame

plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams["figure.titlesize"] = "x-large"


class App(Tk):
    FREQ_SAMPLING = 200e6
    FREQ_CUT = 13e6
    W_CUT = FREQ_CUT / (FREQ_SAMPLING / 2)
    ORDER = 4
    B, A, *_ = signal.butter(ORDER, W_CUT, btype="highpass", output="ba")

    def __init__(self, loop, interval=1 / 120):
        super().__init__()
        self.frames = MainFrame(self)
        self.loop = loop
        self.tasks = []
        self.tasks.append(loop.create_task(self.updater(interval)))

        self.request = Request()
        self.loadpath = ""
        self.savepath = ""
        self.parser = Parser()
        self.handler = Handler()

        self.request_parsed = False
        self.prepare_update = False
        self.acquisition_update = False
        self.correlation_update = False

        self.chunk = None
        self.mean = None
        self.spectrum = None
        self.freq = None
        self.trace = None
        self.traces = None

        self.key = None
        self.cor = None
        self.guess = None
        self.maxs = None
        self.exacts = None
        self.max_env = None
        self.min_env = None
        self.maxs_graph = None
        self.maxs_list = None

        self.i = 0
        self.j = 0

        self.thread_com = None

        self.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        for task in self.tasks:
            task.cancel()
        self.loop.stop()
        super().close()

    async def updater(self, interval):
        while True:
            self.update()
            if self.frames.launched:
                self.tasks.append(self.loop.create_task(self.request_parser()))
                self.frames.launched = False

            if self.request_parsed:
                self.tasks.append(self.loop.create_task(self.launcher()))
                self.request_parsed = False

            if self.prepare_update:
                self.tasks.append(self.loop.create_task(self.update_prepare_acquire()))
                self.prepare_update = False

            if self.acquisition_update:
                print("is okay")
                annotation = f"{'samples':<16}{self.traces.shape[1]}\n" \
                             f"{self.request}\n" \
                             f"{self.parser.meta}"
                self.tasks.append(self.loop.create_task(self.frames.plot.acquisition.draw(
                    self.mean,
                    self.spectrum,
                    self.freq,
                    annotation)))
                self.acquisition_update = False

            if self.correlation_update:
                annotation = f"imported: {self.handler.iterations}\n" \
                             f"guess correlation: {100 * self.maxs[self.i, self.j, self.guess[self.i, self.j]]:.2f}%\n" \
                             f"key correlation: {100 * self.maxs[self.i, self.j, self.key[self.i, self.j]]:.2f}%\n" \
                             f"{self.request}"
                self.tasks.append(self.loop.create_task(self.frames.plot.correlation.update_scale(
                    self.handler,
                    self.request)))
                self.tasks.append(self.loop.create_task(self.frames.plot.correlation.draw(
                    self.i,
                    self.j,
                    self.key,
                    self.cor,
                    self.guess,
                    self.maxs_graph,
                    self.exacts,
                    self.max_env,
                    self.min_env,
                    annotation
                )))
                self.correlation_update = False
            await asyncio.sleep(interval)

    async def launcher(self):
        self.frames.log.log(f"*** Starting side-channel analysis ***\n")
        print(f"{self.request}\n{'save path':<16}{os.path.abspath(self.savepath)}")
        try:
            self.thread_com = threading.Thread(target=ui.actions.acquire,
                                               args=(self.request, self.process, self.prepare, self.savepath))
            self.thread_com.setName("com.uart")
            self.thread_com.start()
        except Exception as e:
            print(e)
            self.frames.log.log(f"{e}\nAcquisition failed due to previous errors...")

    async def request_parser(self):
        self.frames.log.clear()
        success = True
        iterations = self.frames.config.general.var_iterations.get()
        try:
            self.request.iterations = int(iterations)
        except ValueError:
            self.frames.log.log(f"invalid iterations value: {iterations}\n")
            success = False
        name = self.frames.config.general.var_target.get()
        if name:
            self.request.name = name
        else:
            self.frames.log.log(f"target's name cannot be null\n")
            success = False
        self.request.mode = self.frames.config.general.frame_mode.var_mode.get()
        self.request.model = self.frames.config.general.frame_model.var_model.get()
        self.request.source = Request.Sources.SERIAL

        start = self.frames.config.perfs.var_start.get()
        try:
            self.request.start = int(start) if start else 0
        except ValueError:
            self.frames.log.log(f"invalid start value: {start}\n")
            success = False

        end = self.frames.config.perfs.var_end.get()
        try:
            self.request.end = int(end) if end else 0
        except ValueError:
            self.frames.log.log(f"invalid end value: {end}\n")
            success = False

        chunks = self.frames.config.perfs.var_chunks.get()
        try:
            self.request.chunks = int(chunks) if chunks else 0
        except ValueError:
            self.frames.log.log(f"invalid chunks value: {chunks}\n")
            success = False

        self.request.verbose = self.frames.config.perfs.var_verbose.get()
        self.request.noise = self.frames.config.perfs.var_noise.get()

        self.savepath = self.frames.config.file.var_path.get()
        try:
            os.mkdir(self.savepath)
        except FileExistsError:
            pass
        except FileNotFoundError:
            self.frames.log.log(f"Unable to create directory: {self.savepath}")
            success = False

        if success:
            self.frames.log.clear()

        self.request_parsed = success

    async def update_prepare_acquire(self):
        now = datetime.now()
        if self.chunk is not None:
            c = self.chunk + 1
            self.frames.log.var_status.set(f"Chunk {c}/{self.request.chunks} started at {now:%Y-%m-%d %H:%M:%S}")
        else:
            self.frames.log.var_status.set(f"Acquisition started at {now:the %d %b %Y at %H:%M:%S}")

    @classmethod
    @ui.actions.timed("acquisition processing", "acquisition processing success!")
    def compute_acquire(cls, queue):
        trace, traces, iterations = queue.get()
        mean = trace / iterations
        spectrum = np.absolute(fft.fft(mean - np.mean(mean)))
        size = spectrum.size
        freq = np.argsort(np.fft.fftfreq(size, 1.0 / App.FREQ_SAMPLING)[:size // 2] / 1e6)
        queue.put((trace, mean, spectrum, freq))

    @classmethod
    @ui.actions.timed("correlation processing", "correlation processing success!")
    def compute_correlate(cls, queue):
        blocks, key, handler, model, traces = queue.get()
        for trace in traces:
            trace[:] = signal.filtfilt(App.B, App.A, trace)
        if handler.iterations > 0:
            handler.accumulate(blocks, traces)
        else:
            handler.clear(traces.shape[1]).set_model(1).accumulate(blocks, traces)
        cor = handler.correlations()
        guess, maxs, exacts = Handler.guess_stats(cor, key)
        max_env, min_env = Handler.guess_envelope(cor)
        queue.put((handler, cor, guess, maxs, exacts, max_env, min_env))

    def prepare(self, chunk):
        now = datetime.now()
        self.chunk = chunk
        if chunk is not None:
            c = chunk + 1
            print(f"{'chunk':<16}{c}/{self.request.chunks}")
            print(f"{'requested':<16}{c * self.request.iterations}/{self.request.iterations * self.request.chunks}")
        print(f"{'started':<16}{now:%Y-%m-%d %H:%M:%S}")
        self.prepare_update = True

    def process(self, x, chunk):
        self.parser.clear()
        self.parser.parse(x, direction=self.request.direction, verbose=self.request.verbose)
        parsed = len(self.parser.channel)
        ui.save(self.request, x,
                self.parser.leak, self.parser.channel, self.parser.meta, self.parser.noise,
                chunk=chunk, path=self.savepath)
        if chunk is not None:
            print(f"{'chunk':<16}{chunk + 1}/{self.request.chunks}")
        print(f"{'size':<16}{ui.sizeof(len(x or []))}")
        print(f"{'parsed':<16}{parsed}/{self.request.iterations}")
        print(f"{'total':<16}{self.parser.meta.iterations}/{(self.request.chunks or 1) * self.request.iterations}")

        if not parsed:
            warn("no traces parsed!\nskipping...")
            return

        print("\n")

        if self.trace is None:
            self.traces = np.array(tr.adjust(self.parser.leak.traces))
            self.trace = np.sum(self.traces, axis=0)
        else:
            self.traces = np.array(tr.adjust(self.parser.leak.traces, len(self.trace)))
            self.trace += np.sum(self.traces, axis=0)

        self.key = aes.key_expansion(aes.words_to_block(self.parser.channel.keys[0]))[10].T
        blocks = np.array([aes.words_to_block(block) for block in self.parser.channel.ciphers])

        acquisition_queue = mp.Queue()
        correlation_queue = mp.Queue()
        acquisition_process = mp.Process(target=App.compute_acquire, args=(acquisition_queue,))
        correlation_process = mp.Process(target=App.compute_correlate, args=(correlation_queue,))
        acquisition_process.name = "acquisition"
        correlation_process.name = "correlation"
        acquisition_process.start()
        print(f"started {acquisition_process}")
        correlation_process.start()
        print(f"started {correlation_process}")
        acquisition_queue.put((self.trace, self.traces, self.parser.meta.iterations))
        correlation_queue.put((blocks, self.key, self.handler, 1, self.traces))

        self.trace, self.mean, self.spectrum, self.freq = acquisition_queue.get()
        acquisition_process.join()
        print(f"ended {acquisition_process}")
        self.acquisition_update = True

        self.handler, self.cor, self.guess, self.maxs, self.exacts, self.max_env, self.min_env = correlation_queue.get()
        correlation_process.join()
        print(f"ended {correlation_process}")

        if self.maxs_list is not None:
            np.append(self.maxs_list, self.maxs)
        else:
            self.maxs_list = np.array([self.maxs])
        self.maxs_graph = np.moveaxis(self.maxs_list, (0, 1, 2, 3), (3, 0, 1, 2))
        self.correlation_update = True
        print(f"exacts: {np.count_nonzero(self.exacts)}/{BLOCK_LEN * BLOCK_LEN}\n{self.exacts}")
        print(f"key:\n{self.key}")
        print(f"guess:\n{self.guess}")

    def stop(self):
        pass


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

        self.launched = False
        self.stopped = False

    def launch(self):
        self.launched = True
        self.stopped = False

    def stop(self):
        self.launched = False
        self.stopped = True


lo = asyncio.get_event_loop()
app = App(lo)
app.title("SCABox Demo")
lo.run_forever()
lo.close()
