import time
import traceback
from enum import Enum, auto, Flag

import serial_asyncio
import asyncio
import multiprocessing as mp
import os
import threading as th
from datetime import datetime, timedelta
from tkinter import *
from warnings import warn

import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, fft

import signal as sgl

import ui
from lib import cpa
from lib.aes import BLOCK_LEN
from lib.cpa import Handler
from lib.data import Request, Parser, Keywords
import lib.traces as tr
from widgets import MainFrame

plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams["figure.titlesize"] = "x-large"
logger_format = '[%(asctime)s | %(processName)s | %(threadName)s] %(message)s'
logging.basicConfig(stream=sys.stdout, format=logger_format, level=logging.INFO, datefmt="%y-%m-%d %H:%M:%S")


class State(Enum):
    IDLE = 0
    LAUNCHED = 1
    STARTED = 2
    ACQUIRED = 3
    WAIT = 4


class Pending(Flag):
    IDLE = auto()
    VALID = auto()
    STARTING = auto()
    LAUNCHING = auto()
    PARSING = auto()
    STATISTICS = auto()
    CORRELATION = auto()
    CHUNK = auto()


def show_error(*args):
    err = traceback.format_exception(*args)
    logging.info(err)


# but this works too


class AppSerial(asyncio.Protocol):
    def __init__(self):
        self.buffer = bytearray()
        self.transport = None
        self.connected = False
        self.done = True
        self.terminator = Keywords.END_ACQ_TAG
        self.t_start = None
        self.t_end = None
        self.iterations = 0
        self.pending = False

    def connection_made(self, transport):
        self.transport = transport
        self.connected = True
        self.done = True
        self.pending = False
        self.iterations = 0
        logging.info(self.transport.serial)

    def connection_lost(self, exc):
        self.connected = False
        self.done = False
        self.transport.serial.close()
        self.transport.loop_main.stop()
        logging.info(self.transport.serial)
        if exc:
            logging.warning(exc)

    def data_received(self, data):
        self.buffer += data
        if self.buffer[-16:].find(Keywords.END_ACQ_TAG) != -1:
            self.t_end = time.perf_counter()
            logging.info(f"received {len(self.buffer)} bytes in {timedelta(seconds=self.t_end - self.t_start)}")
            self.done = True
            self.pending = True
        if data.find(Keywords.START_TRACE_TAG) != -1:
            self.iterations += 1
            self.pending = True

    async def send(self, buffer):
        logging.info(f"sending {buffer} to {self.transport.serial.name}")
        self.buffer.clear()
        self.transport.serial.flush()
        self.done = False
        self.pending = False
        self.iterations = 0
        self.t_start = time.perf_counter()
        self.transport.serial.write(buffer + b"\r\n")


class App(Tk):
    def __init__(self, loop, interval=1 / 60):
        super().__init__()
        self.frames = MainFrame(self)
        self.loop_main = loop
        self.loop_com = None

        self.tasks = []
        self.tasks.append(loop.create_task(self.event_loop(interval), name="event.update"))

        self.queue_stats = mp.Queue(1)
        self.queue_corr = mp.Queue(1)
        self.queue_comp = mp.Queue(1)
        self.queue_comm = mp.Queue(1)
        self.process_stats = mp.Process(target=App.statistics, args=(self.queue_stats,), name="p.statistics")
        self.process_corr = mp.Process(target=App.correlation, args=(self.queue_corr,), name="p.correlation")
        self.thread_comp = th.Thread(target=self.computation, name="t.computation")
        self.thread_comm = th.Thread(target=self.communication, name="t.communication")

        self.command = ""
        self.request = Request()
        self.parser = Parser()
        self.handler = Handler(0)

        self.state = State.IDLE
        self.pending = Pending.IDLE

        self.curr_chunk = None
        self.next_chunk = None

        self.mean = None
        self.spectrum = None
        self.freq = None
        self.trace = None
        self.traces = None

        self.key = None
        self.corr = None
        self.guess = None
        self.maxs = None
        self.exacts = None
        self.max_env = None
        self.min_env = None
        self.maxs_graph = None
        self.maxs_list = None

        self.serial_transport = None
        self.serial_protocol = None

        self.t_start = None
        self.t_end = None

        self.i = 0
        self.j = 0

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.process_stats.start()
        logging.info(self.process_stats)
        self.process_corr.start()
        logging.info(self.process_corr)
        self.thread_comp.start()
        logging.info(self.thread_comp)
        self.thread_comm.start()
        logging.info(self.thread_comm)
        self.frames.log.log("*** Welcome to SCABox demo ***\n")

    def close(self):
        if self.serial_transport is not None:
            self.serial_transport.loop_main.call_soon_threadsafe(self.serial_transport.close)
        self.queue_comm.put(False)
        self.thread_comm.join()
        self.queue_comm.close()
        logging.info(self.thread_comm)

        for task in self.tasks:
            task.cancel()
        self.loop_main.call_soon_threadsafe(self.loop_main.stop)

        self.queue_stats.put((None,))
        self.process_stats.join()
        self.queue_stats.close()
        logging.info(self.process_stats)

        self.queue_corr.put((None,))
        self.process_corr.join()
        self.queue_corr.close()
        logging.info(self.process_corr)

        self.queue_comp.put(False)
        self.thread_comp.join()
        self.queue_comp.close()
        logging.info(self.thread_comp)

        self.destroy()

    def communication(self):
        self.loop_com = asyncio.new_event_loop()
        while True:
            try:
                if not self.queue_comm.get():
                    return
            except KeyboardInterrupt:
                return

            if not self.serial_transport or self.serial_transport.is_closing():
                try:
                    coro = serial_asyncio.create_serial_connection(
                        self.loop_com,
                        AppSerial,
                        self.request.target,
                        baudrate=921600,
                        timeout=10)
                    self.serial_transport, self.serial_protocol = self.loop_com.run_until_complete(coro)
                    self.loop_com.run_forever()
                except Exception as e:
                    logging.info(e)
                    continue

    def computation(self):
        while True:
            try:
                if not self.queue_comp.get():
                    return
            except KeyboardInterrupt:
                return

            t_start = time.perf_counter()
            self.parser.clear()
            self.parser.parse(self.serial_protocol.buffer,
                              direction=self.request.direction,
                              verbose=self.request.verbose,
                              warns=True)
            t_end = time.perf_counter()
            self.pending |= Pending.PARSING

            parsed = len(self.parser.channel)
            if not parsed:
                logging.critical("no traces parsed, skipping...")
                return
            else:
                logging.info(f"{parsed} traces parsed in {timedelta(seconds=t_end - t_start)}")

            if self.trace is None:
                self.traces = np.array(tr.adjust(self.parser.leak.traces))
                self.trace = np.sum(self.traces, axis=0)
            else:
                self.traces = np.array(tr.adjust(self.parser.leak.traces, len(self.trace)))
                self.trace += np.sum(self.traces, axis=0)

            self.queue_stats.put((self.trace, self.traces, self.parser.meta.iterations))
            self.queue_corr.put((self.handler, self.request.model, self.parser.channel, self.traces, self.maxs_list))

            self.mean, self.spectrum, self.freq = self.queue_stats.get()
            self.pending |= Pending.STATISTICS

            self.handler, self.corr, self.guess, self.maxs, self.exacts, self.max_env, self.min_env, self.maxs_list, self.maxs_graph = self.queue_corr.get()
            self.pending |= Pending.CORRELATION
            if self.curr_chunk is not None:
                self.pending |= Pending.CHUNK
            self.t_end = time.perf_counter()
            logging.info(f"acquisition succeeded in {timedelta(seconds=self.t_end - self.t_start)}")

    @classmethod
    def statistics(cls, queue):
        while True:
            try:
                trace, traces, iterations = queue.get()
            except (ValueError, KeyboardInterrupt):
                return

            t_start = time.perf_counter()
            mean = trace / iterations
            spectrum = np.absolute(fft.fft(mean - np.mean(mean)))
            size = spectrum.size
            freq = np.argsort(np.fft.fftfreq(size, 1.0 / 200e6)[:size // 2] / 1e6)
            queue.put((mean, spectrum, freq))
            t_end = time.perf_counter()
            logging.info(f"{iterations} traces computed in {timedelta(seconds=t_end - t_start)}")

    @classmethod
    def correlation(cls, queue):
        f_sampling = 200e6
        f_nyquist = f_sampling / 2
        f_cut = 13e6
        w_cut = f_cut / f_nyquist
        order = 4
        b, a, *_ = signal.butter(order, w_cut, btype="highpass", output="ba")

        while True:
            try:
                handler, model, channel, traces, maxs_list = queue.get()
            except (ValueError, KeyboardInterrupt):
                return

            t_start = time.perf_counter()
            for trace in traces:
                trace[:] = signal.filtfilt(b, a, trace)
            if handler.iterations > 0:
                handler.set_blocks(channel).accumulate(traces)
            else:
                handler = Handler(model, channel, traces)
                maxs_list = None
            cor = handler.correlations()
            guess, maxs, exacts = Handler.guess_stats(cor, handler.key)
            max_env, min_env = Handler.guess_envelope(cor)
            if maxs_list is not None:
                maxs_list.append(maxs)
            else:
                maxs_list = [maxs]
            maxs_graph = np.moveaxis(np.array(maxs_list), (0, 1, 2, 3), (3, 0, 1, 2))
            t_end = time.perf_counter()
            logging.info(f"{len(channel)} traces computed in {timedelta(seconds=t_end - t_start)}")
            queue.put((handler, cor, guess, maxs, exacts, max_env, min_env, maxs_list, maxs_graph))

    async def event_loop(self, interval):
        while True:
            self.update()

            if self.state == State.IDLE:
                self.pending |= Pending.IDLE
                self.pending &= ~Pending.VALID
                if self.validate():
                    self.pending |= Pending.VALID
                    if self.frames.clicked_launch:
                        self.frames.clicked_launch = False
                        self.state = State.LAUNCHED
                        await self.launch()

            elif self.state == State.LAUNCHED:
                if self.serial_protocol and self.serial_protocol.connected and self.serial_protocol.done:
                    if not self.curr_chunk or self.next_chunk - self.curr_chunk == 1:
                        self.state = State.STARTED
                        await self.start()

            elif self.state == State.STARTED:
                if self.serial_protocol.done:
                    self.state = State.ACQUIRED

            elif self.state == State.ACQUIRED:
                if self.curr_chunk is not None and self.next_chunk < self.request.chunks - 1:
                    self.next_chunk += 1
                    self.state = State.LAUNCHED
                else:
                    self.state = State.IDLE
                self.queue_comp.put(True)

            if self.serial_protocol and self.serial_protocol.pending:
                self.serial_protocol.pending = False
                await self.show_serial()

            if self.pending & Pending.IDLE:
                self.pending &= ~ Pending.IDLE
                await self.show_idle()

            if self.pending & Pending.LAUNCHING:
                self.pending &= ~ Pending.LAUNCHING
                await self.show_launching()

            if self.pending & Pending.STARTING:
                self.pending &= ~ Pending.STARTING
                await self.show_starting()

            if self.pending & Pending.PARSING:
                self.pending &= ~ Pending.PARSING
                await self.show_parsing()

            if self.pending & Pending.STATISTICS:
                self.pending &= ~ Pending.STATISTICS
                await self.show_stats()

            if self.pending & Pending.CORRELATION:
                self.pending &= ~ Pending.CORRELATION
                await self.show_corr()

            if self.pending & Pending.CHUNK:
                self.pending &= ~ Pending.CHUNK
                self.curr_chunk += 1

            await asyncio.sleep(interval)

    def validate(self):
        if not self.frames.config.validate():
            return False
        self.request.iterations = self.frames.config.general.iterations or self.request.iterations
        self.request.target = self.frames.config.general.target or self.request.target
        self.request.path = self.frames.config.file.path or self.request.path
        self.request.source = Request.Sources.SERIAL
        self.request.mode = self.frames.config.general.frame_mode.mode
        self.request.model = self.frames.config.general.frame_model.model
        self.request.start = self.frames.config.perfs.start
        self.request.end = self.frames.config.perfs.end
        self.request.chunks = self.frames.config.perfs.chunks
        return True

    async def launch(self):
        self.handler.clear().set_model(self.request.model)
        self.trace = None
        self.curr_chunk = 0 if self.request.chunks else None
        self.next_chunk = 0 if self.request.chunks else None

        if self.serial_transport and self.serial_transport.serial and self.serial_transport.serial.port != self.request.target:
            self.serial_transport.loop_main.call_soon_threadsafe(self.serial_transport.close)

        if not self.serial_protocol or not self.serial_protocol.connected:
            self.queue_comm.put(True)

        self.pending |= Pending.LAUNCHING

    async def start(self):
        self.t_start = time.perf_counter()
        self.command = f"{self.request.command('sca')}"
        await self.serial_protocol.send(self.command.encode())
        self.pending |= Pending.STARTING

    async def stop(self):
        pass

    async def show_idle(self):
        if self.pending & Pending.VALID:
            self.frames.log.var_status.set("Ready to launch acquisition !")
        else:
            self.frames.log.var_status.set("Please correct errors before launching acquisition...")

    async def show_serial(self):
        msg = f"{'acquired':<16}{self.serial_protocol.iterations}/{self.request.iterations}"
        try:
            self.frames.log.overwrite_at_least(msg)
        except IndexError:
            self.frames.log.insert_at_least(msg)

    async def show_launching(self):
        self.frames.plot.clear()
        self.frames.log.clear()
        self.frames.log.log(f"* Attack launched *\n"
                            f"{self.request}\n"
                            f"connecting to target...\n")

    async def show_starting(self):
        now = datetime.now()
        self.frames.log.clear()
        self.frames.log.log(f"* Acquisition started *\n"
                            f"{'command':<16}{self.command}\n")
        if self.curr_chunk is not None:
            self.frames.log.log(f"{'requested':<16}{self.request.requested(self.next_chunk)}/{self.request.total}\n")
            self.frames.log.log(f"{'chunk':<16}{self.next_chunk + 1}/{self.request.chunks}\n")
            self.frames.log.var_status.set(
                f"Chunk {self.next_chunk + 1}/{self.request.chunks} started {now:the %d %b %Y at %H:%M:%S}")
        else:
            self.frames.log.var_status.set(f"Acquisition started {now:the %d %b %Y at %H:%M:%S}")

        self.frames.log.insert_at_least(f"waiting target's answer...\n")

    async def show_parsing(self):

        now = datetime.now()
        parsed = len(self.parser.channel)
        self.frames.log.clear()
        self.frames.log.log(
            f"* Traces parsed *\n"
            f"{'size':<16}{ui.sizeof(len(self.serial_protocol.buffer))}\n"
            f"{'parsed':<16}{parsed}/{self.request.iterations}\n")
        if self.curr_chunk is not None:
            self.frames.log.log(
                f"{'chunk':<16}{self.curr_chunk + 1}/{self.request.chunks}\n"
                f"{'total':<16}{self.handler.iterations + parsed}/{self.request.total}\n")
            self.frames.log.var_status.set(
                f"Chunk {self.curr_chunk + 1}/{self.request.chunks} processing started {now:the %d %b %Y at %H:%M:%S}"
            )
        else:
            self.frames.log.var_status.set(f"Processing started {now:the %d %b %Y at %H:%M:%S}")

    async def show_stats(self):
        now = datetime.now()
        annotation = f"{'samples':<16}{self.traces.shape[1]}\n" \
                     f"{self.request}\n" \
                     f"{self.parser.meta}"
        msg = f"Chunk {self.curr_chunk + 1}/{self.request.chunks} statistics computed {now:the %d %b %Y at %H:%M:%S}" \
            if self.curr_chunk is not None else f"Statistics computed {now:the %d %b %Y at %H:%M:%S}"
        self.frames.log.var_status.set(msg)

        self.frames.plot.acquisition.draw(
            self.mean,
            self.spectrum,
            self.freq,
            "")

    async def show_corr(self):
        now = datetime.now()
        annotation = f"imported: {self.handler.iterations}\n" \
                     f"guess correlation: {100 * self.maxs[self.i, self.j, self.guess[self.i, self.j]]:.2f}%\n" \
                     f"key correlation: {100 * self.maxs[self.i, self.j, self.handler.key[self.i, self.j]]:.2f}%\n" \
                     f"{self.request}"
        msg = f"Chunk {self.curr_chunk + 1}/{self.request.chunks} correlation computed {now:the %d %b %Y at %H:%M:%S}" \
            if self.curr_chunk is not None else f"Correlation computed {now:the %d %b %Y at %H:%M:%S}"
        self.frames.log.var_status.set(msg)
        self.frames.log.clear()
        self.frames.log.log(f"exacts: {np.count_nonzero(self.exacts)}/{BLOCK_LEN * BLOCK_LEN}\n{self.exacts}\n"
                            f"key:\n{self.handler.key}\n"
                            f"guess:\n{self.guess}\n")
        self.frames.plot.correlation.update_scale(
            self.handler,
            self.request)
        self.frames.plot.correlation.draw(
            self.i,
            self.j,
            self.handler.key,
            self.corr,
            self.guess,
            self.maxs_graph,
            self.exacts,
            self.max_env,
            self.min_env,
            "")


if __name__ == "__main__":
    lo = asyncio.get_event_loop()
    app = App(lo)
    app.title("SCABox Demo")
    try:
        lo.run_forever()
    except KeyboardInterrupt:
        app.close()

    lo.close()
