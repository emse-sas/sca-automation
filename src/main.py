import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import threading as th
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum, auto, Flag
from tkinter import *

import numpy as np
import serial_asyncio
from scipy import fft, signal

import lib.traces as tr
from lib.aes import BLOCK_LEN
from lib.cpa import Handler, Statistics
from lib.data import Request, Parser, Keywords
from widgets import MainFrame, config, sizeof
from widgets.config import PlotFrame

logger_format = '[%(asctime)s | %(processName)s | %(threadName)s] %(message)s'
logging.basicConfig(stream=sys.stdout, format=logger_format, level=logging.DEBUG, datefmt="%y-%m-%d %H:%M:%S")


class State(Enum):
    IDLE = 0
    LAUNCHED = 1
    STARTED = 2
    ACQUIRED = 3


class Status(Flag):
    IDLE = auto()
    PAUSE = auto()
    VALID = auto()
    DONE = auto()


class Pending(Flag):
    IDLE = auto()
    VALID = auto()
    STARTING = auto()
    CONNECTING = auto()
    COMPUTING = auto()
    LAUNCHING = auto()
    PARSING = auto()
    STATISTICS = auto()
    CORRELATION = auto()
    CHUNK = auto()
    DONE = auto()
    STOP = auto()
    RESUME = auto()


def show_error(*args):
    err = traceback.format_exception(*args)
    logging.info(err)


# but this works too


class AppSerial(asyncio.Protocol):
    def __init__(self):
        self.buffer = bytes()
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
        self.buffer = bytes()
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
        self.buffer = bytes()
        self.transport.serial.flush()
        self.done = False
        self.pending = False
        self.iterations = 0
        self.t_start = time.perf_counter()

        self.transport.serial.write(buffer + b"\r\n")


class App(Tk):
    def __init__(self, loop, request, interval=1 / 60):
        super().__init__()
        self.frames = MainFrame(self)
        self.loop_main = loop
        self.loop_com = None

        self.tasks = []
        self.tasks.append(loop.create_task(self.event_loop(interval), name="event.update"))
        self.thread_comp = None
        self.thread_comm = None

        self.command = ""
        self.request = request or Request()
        self.parser = Parser()
        self.handler = Handler(Handler.Models.SBOX_R0)
        self.stats = Statistics()
        self.iterations = 0
        self.buffer = None

        self.trace_mean = None
        self.trace_spectrum = None
        self.trace_freq = None
        self.trace_sum = None
        self.traces = None

        self.noise_mean = None
        self.noise_spectrum = None
        self.noise_freq = None
        self.noise_sum = None
        self.noises = None

        self.state = State.IDLE
        self.pending = Pending.IDLE
        self.status = Status.IDLE

        self.curr_chunk = None
        self.next_chunk = None
        self.total_size = 0
        self.size = 0

        self.serial_transport = None
        self.serial_protocol = None

        self.t_start = None
        self.t_end = None

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.frames.log.log("*** Welcome to SCABox demo ***\n")

        if request:
            self._set()
        self.frames.config.target.refresh()

    def close(self):
        if self.serial_transport is not None and self.serial_transport.loop:
            self.serial_transport.loop.call_soon_threadsafe(self.serial_transport.close)

        for task in self.tasks:
            task.cancel()
        self.loop_main.call_soon_threadsafe(self.loop_main.stop)

        if self.thread_comm is not None:
            self.thread_comm.join()
            logging.info(self.thread_comm)

        if self.thread_comp is not None:
            self.thread_comp.join()
            logging.info(self.thread_comp)

        self.destroy()

    def _communication(self):
        self.loop_com = asyncio.new_event_loop()
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
            except Exception as err:
                logging.info(err)
        self.loop_com.close()

    def _computation(self):
        t_start = time.perf_counter()
        if not self.parse():
            return
        self.pending |= Pending.PARSING

        if not self.save():
            pass

        pipe_stats = mp.Pipe()
        pipe_noise = mp.Pipe()
        pipe_corr = mp.Pipe()

        process_stats = mp.Process(target=self.statistics, args=(self.trace_sum, self.parser.leak, pipe_stats[1]),
                                   name="p.stats")
        process_noise = mp.Process(target=self.statistics, args=(self.noise_sum, self.parser.noise, pipe_noise[1]),
                                   name="p.noise")
        process_corr = mp.Process(target=self.correlation, args=(pipe_corr[1],),
                                  name="p.corr")
        process_stats.start()
        if self.request.noise:
            process_noise.start()

        self.traces = pipe_stats[0].recv()
        self.noises = pipe_noise[0].recv() if self.request.noise else None
        self.trace_sum = pipe_stats[0].recv()
        self.noise_sum = pipe_noise[0].recv() if self.request.noise else None
        self.trace_mean = pipe_stats[0].recv()
        self.noise_mean = pipe_noise[0].recv() if self.request.noise else None

        process_corr.start()
        self.trace_spectrum = pipe_stats[0].recv()
        self.noise_spectrum = pipe_noise[0].recv() if self.request.noise else None
        self.trace_freq = pipe_stats[0].recv()
        self.noise_freq = pipe_noise[0].recv() if self.request.noise else None

        if self.frames.config.plot.mode == config.PlotFrame.Mode.STATISTICS \
                or self.frames.config.plot.mode == config.PlotFrame.Mode.NOISE:
            self.pending |= Pending.STATISTICS

        self.handler = pipe_corr[0].recv()
        self.stats = pipe_corr[0].recv()
        self.frames.plot.update_scale(self.handler, self.request)
        if self.frames.config.plot.mode == config.PlotFrame.Mode.CORRELATION:
            self.pending |= Pending.CORRELATION

        process_stats.join()
        if self.request.noise:
            process_noise.join()
        process_corr.join()

        if self.curr_chunk is not None:
            self.pending |= Pending.CHUNK
        self.t_end = time.perf_counter()
        t_end = time.perf_counter()
        logging.info(f"computing succeeded in {timedelta(seconds=t_end - t_start)}")
        logging.info(f"acquisition succeeded in {timedelta(seconds=self.t_end - self.t_start)}")

    def parse(self):
        t_start = time.perf_counter()
        self.parser.clear()
        self.parser.parse(self.buffer,
                          direction=self.request.direction,
                          verbose=self.request.verbose,
                          noise=self.request.noise,
                          warns=True)
        t_end = time.perf_counter()
        parsed = len(self.parser.channel)
        if not parsed:
            logging.critical("no traces parsed, skipping...")
            if self.curr_chunk is not None:
                self.pending |= Pending.CHUNK
            self.t_end = time.perf_counter()
            return False
        else:
            logging.info(f"{parsed} traces parsed in {timedelta(seconds=t_end - t_start)}")
        return True

    def save(self):
        try:
            append = self.request.chunks is not None
            suffix = f"_{self.curr_chunk}.bin" if self.curr_chunk is not None else ".bin"
            path = self.request.path
            with open(os.path.join(self.request.path, self.request.filename(suffix=suffix)), "wb+") as file:
                file.write(self.serial_protocol.buffer)
            self.parser.channel.write_csv(os.path.join(path, self.request.filename("channel", ".csv")), append)
            self.parser.leak.write_csv(os.path.join(path, self.request.filename("leak", ".csv")), append)
            self.parser.meta.write_csv(os.path.join(path, self.request.filename("meta", ".csv")), append)
            self.parser.noise.write_csv(os.path.join(path, self.request.filename("noise", ".csv")), append)
            logging.info(f"traces successfully saved {(self.curr_chunk or 0) + 1}/{self.request.chunks or 1}")
        except OSError as err:
            logging.error(f"error occurred during saving: {err}")
            return False
        return True

    def statistics(self, trace, leak, pipe):
        t_start = time.perf_counter()
        traces = np.array(tr.adjust(leak.traces, None if trace is None else trace.shape[0]))
        pipe.send(traces)
        if trace is None:
            trace = np.sum(traces, axis=0)
        else:
            trace += np.sum(traces, axis=0)
        pipe.send(trace)
        mean = np.divide(trace, self.iterations)
        pipe.send(mean)
        spectrum = np.absolute(fft.fft(mean - np.mean(mean)))
        pipe.send(spectrum)
        size = spectrum.size
        freq = np.argsort(np.fft.fftfreq(size, 1.0 / 200e6)[:size // 2] / 1e6)
        pipe.send(freq)
        t_end = time.perf_counter()
        logging.info(f"{self.iterations} traces computed in {timedelta(seconds=t_end - t_start)}")

    def correlation(self, pipe):
        t_start = time.perf_counter()
        if self.request.noise:
            noise_mean = np.mean(self.noises, axis=0)
            noise_mean -= np.mean(noise_mean)
            noise_mean = signal.detrend(noise_mean)
            noise_spectrum = fft.fft(noise_mean)
            for trace in self.traces:
                trace = signal.detrend(trace)
                trace_spectrum = fft.fft(trace - np.mean(trace))
                trace[:] = fft.ifft(trace_spectrum - noise_spectrum)
                # _, trace[:] = signal.deconvolve(trace - np.mean(trace), noise_mean)
        if self.handler.iterations > 0:
            self.handler.set_blocks(self.parser.channel).accumulate(self.traces)
        else:
            self.handler = Handler(self.request.model, self.parser.channel, self.traces)
        pipe.send(self.handler)
        self.stats.update(self.handler)
        pipe.send(self.stats)
        t_end = time.perf_counter()
        logging.info(f"{self.iterations} traces computed in {timedelta(seconds=t_end - t_start)}")

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

    async def update_state(self):
        if self.frames.config.plot.validate() and self.frames.config.plot.changed and self.handler.iterations > 0:
            if self.frames.config.plot.mode == config.PlotFrame.Mode.CORRELATION:
                self.pending |= Pending.CORRELATION
            elif self.frames.config.plot.mode == config.PlotFrame.Mode.STATISTICS:
                self.pending |= Pending.STATISTICS
            elif self.frames.config.plot.mode == config.PlotFrame.Mode.NOISE:
                self.pending |= Pending.STATISTICS

        if self.state != State.IDLE:
            if self.frames.clicked_stop and self.status & Status.PAUSE:
                self.state = State.IDLE
                self.pending |= Pending.STOP
                self.status &= ~Status.PAUSE
            elif self.frames.clicked_stop:
                self.pending |= Pending.STOP
                self.status |= Status.PAUSE
                await self.stop()
            elif self.frames.clicked_launch and self.status & Status.PAUSE:
                self.pending |= Pending.RESUME
                self.status &= ~Status.PAUSE

        if self.state == State.IDLE:
            try:
                valid = self._validate()
            except TclError as err:
                logging.warning(f"error occurred during validation {err}")
                valid = False

            self.pending |= Pending.IDLE
            self.status = self.status | Status.VALID if valid else self.status & ~Status.VALID
            if self.frames.clicked_launch and valid:
                self.state = State.LAUNCHED
                self.pending |= Pending.LAUNCHING
                await self.launch()

        elif self.state == State.LAUNCHED:
            if self.status & Status.PAUSE:
                self.state = State.IDLE
                self.pending |= Pending.STOP
                self.status &= ~Status.PAUSE
            elif self.serial_protocol and self.serial_protocol.connected:
                if not self.next_chunk or (self.next_chunk - self.curr_chunk <= 1 and self.serial_protocol.done):
                    self.state = State.STARTED
                    self.pending |= Pending.STARTING | Pending.RESUME
                    await self.start()
            else:
                self.pending |= Pending.CONNECTING

        elif self.state == State.STARTED:
            if not self.serial_protocol.connected:
                self.state = State.LAUNCHED
                self.pending |= Pending.CONNECTING | Pending.LAUNCHING
                return

            if self.status & Status.PAUSE and not self.serial_protocol.paused:
                self.serial_protocol.pause_reading()
            elif ~self.status & Status.PAUSE and self.serial_protocol.paused:
                self.serial_protocol.resume_reading()

            if not self.serial_protocol.paused and self.serial_protocol.done:
                self.state = State.ACQUIRED

        elif self.state == State.ACQUIRED:
            self.iterations = self.serial_protocol.total_iterations
            self.buffer = bytes(self.serial_protocol.buffer)
            self.pending |= Pending.COMPUTING
            if self.curr_chunk is not None and self.next_chunk < self.request.chunks - 1:
                self.next_chunk += 1
                self.state = State.LAUNCHED
            else:
                self.state = State.IDLE
                self.pending |= Pending.DONE
            if self.thread_comp is None:
                self.thread_comp = th.Thread(target=self._computation, name="t.comp")
                self.thread_comp.start()

    async def acknowledge_pending(self):
        if self.frames.clicked_refresh:
            self.frames.clicked_refresh = False
            self.frames.config.target.refresh()

        if self.thread_comp is not None:
            self.thread_comp.join(timeout=0)
            if not self.thread_comp.is_alive():
                self.thread_comp = None

        if self.thread_comm is not None:
            self.thread_comm.join(timeout=0)
            if not self.thread_comm.is_alive():
                self.thread_comm = None

        if self.serial_protocol and self.serial_protocol.pending:
            self.serial_protocol.pending = False
            await self.show_serial()

        if self.pending & Pending.IDLE:
            self.pending &= ~ Pending.IDLE
            self.command = f"{self.request.command('sca')}"
            await self.show_idle()

        if self.pending & Pending.LAUNCHING:
            self.pending &= ~ Pending.LAUNCHING
            if not self.curr_chunk:
                self.frames.plot.clear()
                self.frames.log.clear()
            self.frames.lock_launch()
            self.frames.config.lock()
            self.frames.button_stop.focus_set()
            await self.show_launching()

        if self.pending & Pending.STARTING:
            self.pending &= ~ Pending.STARTING
            await self.show_starting()

        if self.pending & Pending.PARSING:
            self.pending &= ~ Pending.PARSING
            await self.show_parsing()

        if self.pending & Pending.CONNECTING:
            if self.thread_comm is None:
                self.pending &= ~ Pending.CONNECTING
                self.thread_comm = th.Thread(target=self._communication, name="t.comm")
                self.thread_comm.start()
                self.frames.config.target.refresh()

        if self.pending & Pending.COMPUTING:
            self.pending &= ~ Pending.COMPUTING

        if self.pending & Pending.CONNECTING:
            self.pending &= ~ Pending.CONNECTING
            self.frames.config.target.refresh()

        if self.pending & Pending.STATISTICS:
            self.pending &= ~ Pending.STATISTICS
            await self.show_stats()

        if self.pending & Pending.CORRELATION:
            self.pending &= ~ Pending.CORRELATION
            self.frames.log.clear()
            await self.show_corr()
            await self.show_parsing()
            if self.state != State.IDLE:
                await self.show_starting()

        if self.pending & Pending.CHUNK:
            self.pending &= ~ Pending.CHUNK
            self.curr_chunk += 1

        if self.pending & Pending.DONE:
            self.frames.config.unlock()
            self.frames.unlock_launch()
            self.pending &= ~ Pending.DONE

        if self.pending & Pending.STOP:
            if self.status & Status.PAUSE:
                self.frames.unlock_launch()
            else:
                self.frames.config.unlock()
            self.pending &= ~ Pending.STOP

        if self.pending & Pending.RESUME:
            self.frames.lock_launch()
            self.pending &= ~Pending.RESUME

    def _validate(self):
        if not self.frames.config.validate():
            return False
        self.request.iterations = self.frames.config.general.iterations or self.request.iterations
        self.request.chunks = self.frames.config.general.chunks
        self.request.mode = self.frames.config.general.mode
        self.request.model = self.frames.config.general.model
        self.request.source = self.frames.config.target.source or self.request.source
        self.request.target = self.frames.config.target.target or self.request.target
        self.request.start = self.frames.config.perfs.format.start
        self.request.end = self.frames.config.perfs.format.end
        self.request.verbose = self.frames.config.perfs.format.verbose
        self.request.noise = self.frames.config.perfs.format.noise
        self.request.path = self.frames.config.file.path or self.request.path
        return True

    def _set(self):
        self.frames.config.general.iterations = self.request.iterations
        self.frames.config.perfs.chunks = self.request.chunks
        self.frames.config.general.mode = self.request.mode
        self.frames.config.general.model = self.request.model
        self.frames.config.target.target = self.request.target
        self.frames.config.perfs.format.start = self.request.start
        self.frames.config.perfs.format.end = self.request.end
        self.frames.config.perfs.format.noise = self.request.noise
        self.frames.config.perfs.format.verbose = self.request.verbose
        self.frames.config.file.path = self.request.path
        return True

    async def launch(self):
        logging.info("launching attack...")
        self.handler.clear().set_model(self.request.model)
        self.parser.clear()
        self.trace_mean = None
        self.trace_spectrum = None
        self.trace_freq = None
        self.trace_sum = None
        self.traces = None

        self.noise_mean = None
        self.noise_spectrum = None
        self.noise_freq = None
        self.noise_sum = None
        self.noises = None
        self.stats.clear()
        self.t_start = time.perf_counter()
        self.curr_chunk = 0 if self.request.chunks else None
        self.next_chunk = 0 if self.request.chunks else None
        try:
            os.makedirs(os.path.abspath(self.request.path))
        except FileExistsError:
            pass

        if self.serial_protocol:
            self.serial_protocol.total_iterations = 0
            self.serial_protocol.total_size = 0
            if self.serial_transport.serial:
                if self.serial_transport.serial.port != self.request.target:
                    self.serial_transport.loop.call_soon_threadsafe(self.serial_transport.close)
                    self.pending |= Pending.CONNECTING
            else:
                self.pending |= Pending.CONNECTING
        else:
            self.pending |= Pending.CONNECTING

    async def start(self):
        self.command = f"{self.request.command('sca')}"
        await self.serial_protocol.send(self.command.encode())
        logging.info(f"starting acquisition {(self.next_chunk or 0) + 1}/{self.request.chunks or 1}")

    async def stop(self):
        self.t_end = time.perf_counter()
        self.t_start = self.t_start or self.t_end
        logging.info(f"stopping acquisition, duration {timedelta(seconds=self.t_start - self.t_end)}")

    async def show_idle(self):
        if self.status & Status.VALID:
            self.frames.log.var_status.set("Ready to launch acquisition !")
            self.frames.unlock_launch()
        else:
            self.frames.log.var_status.set("Please correct errors before launching acquisition...")
            self.frames.lock_launch()

        msg = f"{'target':<16}{self.request.target}\n"
        msg += f"{'requested':<16}{self.request.total}\n"
        msg += f"{'command':<16}{self.command}\n"
        self.frames.log.update_text_status(msg)

    async def show_serial(self):
        acquired = self.serial_protocol.iterations
        msg = f"{'acquired':<16}{acquired}/{self.request.iterations}\n"
        t = time.perf_counter()
        if self.curr_chunk is not None:
            msg += f"{'requested':<16}{self.request.requested(self.next_chunk) + acquired}/{self.request.total}\n"
        msg += f"{'speed':<16}{self.serial_protocol.total_iterations / (t - self.t_start):3.1f} i/s\n" \
               f"{'elapsed':<16}{timedelta(seconds=int(t) - int(self.t_start))}\n"
        self.frames.log.update_text_status(msg)

    async def show_launching(self):
        self.frames.log.log(f"* Attack launched *\n"
                            f"{self.request}\n"
                            f"connecting to target...\n")

    async def show_starting(self):
        now = f"{datetime.now():the %d %b %Y at %H:%M:%S}"
        self.frames.log.log(f"* Acquisition started *\n"
                            f"{'command':<16}{self.command}\n")
        if self.curr_chunk is not None:
            self.frames.log.log(f"{'chunk':<16}{self.next_chunk + 1}/{self.request.chunks}\n")
            self.frames.log.var_status.set(f"Chunk {self.next_chunk + 1}/{self.request.chunks} started {now}")
        else:
            self.frames.log.var_status.set(f"Acquisition started {now}")

    async def show_parsing(self):
        now = f"{datetime.now():the %d %b %Y at %H:%M:%S}"
        parsed = len(self.parser.channel)
        self.frames.log.log(
            f"* Traces parsed *\n"
            f"{'parsed':<16}{parsed}/{self.request.iterations}\n"
            f"{'size':<16}{sizeof(self.serial_protocol.size):<8}/{sizeof(self.serial_protocol.total_size):<8}\n")
        if self.curr_chunk is not None:
            self.frames.log.log(
                f"{'chunk':<16}{self.curr_chunk + 1}/{self.request.chunks}\n"
                f"{'total':<16}{self.handler.iterations}/{self.request.total}\n")
            self.frames.log.var_status.set(
                f"Chunk {self.curr_chunk + 1}/{self.request.chunks} processing started {now}"
            )
        else:
            self.frames.log.var_status.set(f"Processing started {now}")

    async def show_stats(self):
        now = f"{datetime.now():the %d %b %Y at %H:%M:%S}"
        if self.curr_chunk is not None:
            msg = f"Chunk {self.curr_chunk + 1}/{self.request.chunks} statistics computed {now}"
        else:
            msg = f"Statistics computed {now}"
        self.frames.log.var_status.set(msg)
        if self.frames.config.plot.mode == PlotFrame.Mode.STATISTICS:
            self.frames.plot.draw_stats((self.trace_mean, self.trace_spectrum, self.trace_freq))
        elif self.frames.config.plot.mode == PlotFrame.Mode.NOISE:
            self.frames.plot.draw_stats((self.noise_mean, self.noise_spectrum, self.noise_freq))
        else:
            raise ValueError(f"invalid plot mode : {self.frames.config.plot.mode}")

    async def show_corr(self):
        now = f"{datetime.now():the %d %b %Y at %H:%M:%S}"
        if self.curr_chunk is not None:
            msg = f"Chunk {self.curr_chunk + 1}/{self.request.chunks} correlation computed {now}"
        else:
            msg = f"Correlation computed {now}"

        self.frames.log.var_status.set(msg)
        self.frames.log.log(f"* Correlation computed *\n"
                            f"{self.stats}\n"
                            f"{'exacts':<16}{np.count_nonzero(self.stats.exacts[-1])}/{BLOCK_LEN * BLOCK_LEN}\n")
        self.frames.plot.draw_corr(self.stats, min(self.frames.config.plot.byte, 15))


argp = argparse.ArgumentParser(description="Side-channel attack demonstration GUI.")
argp.add_argument("-i", "--iterations", type=int,
                  help="Requested count of traces.")
argp.add_argument("-c", "--chunks", type=int, default=None,
                  help="Count of chunks to acquire.")
argp.add_argument("-m", "--mode",
                  choices=[e.name for e in Request.Modes],
                  default=Request.Modes.HARDWARE.name,
                  help="Encryption mode.")
argp.add_argument("--model", choices=[e.name for e in Handler.Models],
                  default=Handler.Models.INV_SBOX_R10.name,
                  help="Leakage model.")
argp.add_argument("-t", "--target", type=str,
                  help="Serial acquisition target name.")

argp.add_argument("--path", type=str,
                  help="Path where to save files.")
argp.add_argument("-p", "--plot", type=int, default=16,
                  help="Count of raw traces to plot.")
argp.add_argument("--start", type=int,
                  help="Start time sample index of each trace.")
argp.add_argument("--end", type=int,
                  help="End time sample index of each trace.")
argp.add_argument("-v", "--verbose", action="store_true",
                  help="Perform verbose serialization during acquisition.")
argp.add_argument("-n", "--noise", action="store_true",
                  help="Acquire noise before starting each synchronous capture.")

if __name__ == "__main__":

    lo = asyncio.get_event_loop()
    app = App(lo, request=Request(argp.parse_args()))
    app.title("SCABox Demo")
    try:
        lo.run_forever()
    except KeyboardInterrupt:
        logging.error(f"keyboard interrupt, exiting...")
        app.close()

    lo.close()
