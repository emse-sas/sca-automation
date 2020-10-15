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
from lib.cpa import Handler, Statistics
from lib.data import Request, Parser, Keywords
import lib.traces as tr
from widgets import MainFrame

plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams["figure.titlesize"] = "x-large"
logger_format = '[%(asctime)s | %(processName)s | %(threadName)s] %(message)s'
logging.basicConfig(stream=sys.stdout, format=logger_format, level=logging.DEBUG, datefmt="%y-%m-%d %H:%M:%S")


class State(Enum):
    IDLE = 0
    LAUNCHED = 1
    STARTED = 2
    ACQUIRED = 3


class Pending(Flag):
    IDLE = auto()
    VALID = auto()
    STARTING = auto()
    CONNECTING = auto()
    LAUNCHING = auto()
    PARSING = auto()
    STATISTICS = auto()
    CORRELATION = auto()
    CHUNK = auto()
    PAUSE = auto()


def show_error(*args):
    err = traceback.format_exception(*args)
    logging.info(err)


# but this works too


class AppSerial(asyncio.Protocol):
    def __init__(self):
        self.buffer = bytearray()
        self.transport = None
        self.connected = False
        self.done = False
        self.pending = False
        self.paused = False
        self.terminator = Keywords.END_ACQ_TAG
        self.t_start = None
        self.t_end = None
        self.iterations = 0
        self.total_iterations = 0
        self.total_size = 0
        self.size = 0

    def connection_made(self, transport):
        self.total_size = 0
        self.size = 0
        self.buffer.clear()
        self.transport = transport
        self.connected = True
        self.done = False
        self.pending = False
        self.paused = False
        self.iterations = 0
        self.total_iterations = 0
        logging.info(self.transport.serial)

    def connection_lost(self, exc):
        self.connected = False
        self.done = False
        self.paused = False
        self.transport.serial.close()
        self.transport.loop.stop()
        logging.info(self.transport.serial)
        if exc:
            logging.warning(exc)

    def pause_reading(self):
        self.transport.pause_reading()
        self.paused = True
        logging.info(self.transport.serial)
        logging.info("reading paused...")

    def resume_reading(self):
        self.transport.resume_reading()
        self.paused = False
        logging.info(self.transport.serial)
        logging.info("resuming reading...")

    def data_received(self, data):
        self.buffer += data
        if self.buffer[-16:].find(Keywords.END_ACQ_TAG) != -1:
            self.t_end = time.perf_counter()
            self.size = len(self.buffer)
            self.total_size += self.size
            logging.info(f"received {self.size} bytes in {timedelta(seconds=self.t_end - self.t_start)}")
            self.done = True
            self.pending = True
        if data.find(Keywords.START_TRACE_TAG) != -1:
            self.iterations += 1
            self.total_iterations += 1
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
        self.stats = Statistics()

        self.state = State.IDLE
        self.pending = Pending.IDLE
        self.paused = False

        self.curr_chunk = None
        self.next_chunk = None
        self.total_size = 0
        self.size = 0

        self.mean = None
        self.spectrum = None
        self.freq = None
        self.trace = None
        self.traces = None
        self.maxs = None

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
        self.queue_comm.put(False)
        if self.serial_transport is not None and self.serial_transport.loop:
            self.serial_transport.loop.call_soon_threadsafe(self.serial_transport.close)
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

            self.queue_stats.put((self.trace, self.traces, self.serial_protocol.total_iterations))
            self.queue_corr.put((self.handler, self.stats, self.request.model, self.parser.channel, self.traces))
            try:
                self.mean, self.spectrum, self.freq = self.queue_stats.get()
            except ValueError:
                return

            self.pending |= Pending.STATISTICS
            try:
                self.handler, self.stats, self.maxs = self.queue_corr.get()
            except ValueError:
                return
            self.pending |= Pending.CORRELATION

            self.frames.plot.correlation.update_scale(
                self.handler,
                self.request)

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
                handler, stats, model, channel, traces = queue.get()
            except (ValueError, KeyboardInterrupt):
                return

            t_start = time.perf_counter()
            for trace in traces:
                trace[:] = signal.filtfilt(b, a, trace)
            if handler.iterations > 0:
                handler.set_blocks(channel).accumulate(traces)
            else:
                handler = Handler(model, channel, traces)
            stats.update(handler)
            maxs = Statistics.graph(stats.maxs)
            t_end = time.perf_counter()
            logging.info(f"{len(channel)} traces computed in {timedelta(seconds=t_end - t_start)}")
            queue.put((handler, stats, maxs))

    async def update_state(self):
        if self.frames.clicked_stop and self.pending & Pending.PAUSE:
            self.state = State.IDLE
        elif self.frames.clicked_stop:
            self.pending |= Pending.PAUSE
            await self.stop()
        elif self.frames.clicked_launch and self.pending & Pending.PAUSE:
            self.pending &= ~Pending.PAUSE

        if self.state == State.IDLE:
            try:
                valid = self._validate()
            except TclError as err:
                logging.warning(f"error occurred during validation {err}")
                valid = False
            self.pending |= Pending.IDLE | (Pending.VALID if valid else Pending.IDLE)
            if self.frames.clicked_launch and valid:
                self.state = State.LAUNCHED
                self.pending |= Pending.LAUNCHING
                await self.launch()

        elif self.state == State.LAUNCHED:
            if self.pending & Pending.PAUSE:
                self.state = State.IDLE
                self.pending &= ~Pending.PAUSE
                return

            if self.serial_protocol and self.serial_protocol.connected:
                if not self.next_chunk or (
                        self.next_chunk - self.curr_chunk <= 1 and self.serial_protocol.done):
                    self.state = State.STARTED
                    self.pending |= Pending.STARTING
                    await self.start()
            else:
                self.pending |= Pending.CONNECTING

        elif self.state == State.STARTED:
            if not self.serial_protocol.connected:
                self.state = State.LAUNCHED
                self.pending |= Pending.CONNECTING
                self.pending |= Pending.LAUNCHING
                return

            if self.pending & Pending.PAUSE and not self.serial_protocol.paused:
                self.serial_protocol.pause_reading()
            elif ~self.pending & Pending.PAUSE and self.serial_protocol.paused:
                self.serial_protocol.resume_reading()

            if not self.serial_protocol.paused and self.serial_protocol.done:
                self.state = State.ACQUIRED

        elif self.state == State.ACQUIRED:
            if self.curr_chunk is not None and self.next_chunk < self.request.chunks - 1:
                self.next_chunk += 1
                self.state = State.LAUNCHED
            else:
                self.state = State.IDLE
            self.queue_comp.put(True)

    async def acknowledge_pending(self):
        if self.serial_protocol and self.serial_protocol.pending:
            self.serial_protocol.pending = False
            await self.show_serial()

        if self.pending & Pending.IDLE:
            self.pending &= ~ Pending.IDLE
            await self.show_idle()

        if self.pending & Pending.LAUNCHING:
            if not self.curr_chunk:
                self.frames.plot.clear()
                self.frames.log.clear()
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
            self.frames.log.clear()
            self.pending &= ~ Pending.CORRELATION
            await self.show_corr()
            await self.show_parsing()
            if self.state != State.IDLE:
                await self.show_starting()

        if self.pending & Pending.CONNECTING:
            self.pending &= ~ Pending.CONNECTING
            self.queue_comm.put(True)

        if self.pending & Pending.CHUNK:
            self.pending &= ~ Pending.CHUNK
            self.curr_chunk += 1

    async def event_loop(self, interval):
        while True:
            try:
                self.update()
                await self.update_state()
                await self.acknowledge_pending()
                self.frames.clicked_launch = False
                self.frames.clicked_stop = False
                await asyncio.sleep(interval)
            except Exception as err:
                logging.error(f"fatal error occurred `{err}`:\n{traceback.format_exc()}")
                self.close()
                return

    def _validate(self):
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
        logging.info("launching attack...")
        self.handler.clear().set_model(self.request.model)
        self.trace = None
        self.maxs = None
        self.stats.clear()
        self.t_start = time.perf_counter()
        self.curr_chunk = 0 if self.request.chunks else None
        self.next_chunk = 0 if self.request.chunks else None
        if self.serial_protocol:
            self.serial_protocol.total_iterations = 0
            self.serial_protocol.total_size = 0
            if self.serial_transport.serial.port != self.request.target:
                self.serial_transport.loop.call_soon_threadsafe(self.serial_transport.close)
                self.pending |= Pending.CONNECTING
        else:
            self.pending |= Pending.CONNECTING

    async def start(self):
        logging.info(f"starting acquisition {(self.next_chunk or 0) + 1}/{self.request.chunks or 1}")
        self.command = f"{self.request.command('sca')}"
        await self.serial_protocol.send(self.command.encode())

    async def stop(self):
        self.t_end = time.perf_counter()
        logging.info(f"stopping acquisition, duration {timedelta(seconds=self.t_start - self.t_end)}")

    async def show_idle(self):
        if self.pending & Pending.VALID:
            self.frames.log.var_status.set("Ready to launch acquisition !")
        else:
            self.frames.log.var_status.set("Please correct errors before launching acquisition...")

    async def show_serial(self):
        acquired = self.serial_protocol.iterations
        msg = f"{'acquired':<16}{acquired}/{self.request.iterations}\n"
        t = time.perf_counter()
        if self.curr_chunk is not None:
            msg += f"{'requested':<16}{self.request.requested(self.next_chunk) + acquired}/{self.request.total}\n"
        msg += f"{'speed':<16}{self.serial_protocol.total_iterations / (t - self.t_start):3.1f} i/s\n" \
               f"{'elapsed':<16}{timedelta(seconds=int(t) - int(self.t_start))}\n"
        try:
            self.frames.log.overwrite_at_least(msg)
        except IndexError:
            self.frames.log.insert_at_least(msg)

    async def show_launching(self):
        self.frames.log.log(f"* Attack launched *\n"
                            f"{self.request}\n"
                            f"connecting to target...\n")

    async def show_starting(self):
        now = datetime.now()
        self.frames.log.log(f"* Acquisition started *\n"
                            f"{'command':<16}{self.command}\n")
        if self.curr_chunk is not None:
            self.frames.log.log(f"{'chunk':<16}{self.next_chunk + 1}/{self.request.chunks}\n")
            self.frames.log.var_status.set(
                f"Chunk {self.next_chunk + 1}/{self.request.chunks} started {now:the %d %b %Y at %H:%M:%S}")
        else:
            self.frames.log.var_status.set(f"Acquisition started {now:the %d %b %Y at %H:%M:%S}")

    async def show_parsing(self):
        now = datetime.now()
        parsed = len(self.parser.channel)
        self.frames.log.log(
            f"* Traces parsed *\n"
            f"{'parsed':<16}{parsed}/{self.request.iterations}\n"
            f"{'size':<16}{ui.sizeof(self.serial_protocol.size)}/{ui.sizeof(self.serial_protocol.total_size)}\n")
        if self.curr_chunk is not None:
            self.frames.log.log(
                f"{'chunk':<16}{self.curr_chunk + 1}/{self.request.chunks}\n"
                f"{'total':<16}{self.handler.iterations}/{self.request.total}\n")
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
        """
                annotation = f"imported: {self.handler.iterations}\n" \
                     f"guess correlation: {100 * self.maxs[self.i, self.j, self.guess[self.i, self.j]]:.2f}%\n" \
                     f"key correlation: {100 * self.maxs[self.i, self.j, self.handler.key[self.i, self.j]]:.2f}%\n" \
                     f"{self.request}"
        """

        msg = f"Chunk {self.curr_chunk + 1}/{self.request.chunks} correlation computed {now:the %d %b %Y at %H:%M:%S}" \
            if self.curr_chunk is not None else f"Correlation computed {now:the %d %b %Y at %H:%M:%S}"
        self.frames.log.var_status.set(msg)
        self.frames.log.log(f"* Correlation computed *\n"
                            f"{'exacts':<16}{np.count_nonzero(self.stats.exacts[-1])}/{BLOCK_LEN * BLOCK_LEN}\n"
                            f"{self.stats}\n")
        self.frames.plot.correlation.draw(
            self.i,
            self.j,
            self.handler.key,
            self.stats.corr,
            self.stats.guesses[-1],
            self.maxs,
            self.stats.exacts[-1],
            self.stats.corr_max,
            self.stats.corr_min,
            "")


if __name__ == "__main__":
    lo = asyncio.get_event_loop()
    app = App(lo)
    app.title("SCABox Demo")
    try:
        lo.run_forever()
    except Exception as e:
        logging.error(e)
        app.close()

    lo.close()
