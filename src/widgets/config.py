import os
from enum import Enum
from tkinter import *
from lib.cpa import Models
from lib.data import Request
import serial.tools.list_ports


def _set_validation_fg(widget, valid):
    widget.config({"foreground": "Green" if valid else "Red"})


def _clear_fg(widget):
    widget.config({"foreground": "Black"})


class ModeFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Mode")
        self._mode = Request.Modes.HARDWARE
        self.var_mode = StringVar(value=self._mode)
        self.radio_mode_hw = Radiobutton(self,
                                         text="Hardware",
                                         variable=self.var_mode,
                                         value=Request.Modes.HARDWARE,
                                         command=self.on_click)
        self.radio_mode_hw.grid(row=0, column=0, sticky=W, padx=4)

        self.radio_mode_tiny = Radiobutton(self,
                                           text="Tiny",
                                           variable=self.var_mode,
                                           value=Request.Modes.TINY,
                                           command=self.on_click)
        self.radio_mode_tiny.grid(row=1, column=0, sticky=W, padx=4)

        self.radio_mode_ssl = Radiobutton(self,
                                          text="OpenSSL",
                                          variable=self.var_mode,
                                          value=Request.Modes.SSL,
                                          command=self.on_click)
        self.radio_mode_ssl.grid(row=2, column=0, sticky=W, padx=4)

        self.radio_mode_dhuertas = Radiobutton(self,
                                               text="Dhuertas",
                                               variable=self.var_mode,
                                               value=Request.Modes.DHUERTAS,
                                               command=self.on_click)
        self.radio_mode_dhuertas.grid(row=3, column=0, sticky=W, padx=4)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode is None:
            return
        self._mode = mode
        self.var_mode.set(mode)

    def on_click(self):
        self._mode = self.var_mode.get()

    def lock(self):
        self.radio_mode_hw["state"] = DISABLED
        self.radio_mode_tiny["state"] = DISABLED
        self.radio_mode_ssl["state"] = DISABLED
        self.radio_mode_dhuertas["state"] = DISABLED

    def unlock(self):
        self.radio_mode_hw["state"] = NORMAL
        self.radio_mode_tiny["state"] = NORMAL
        self.radio_mode_ssl["state"] = NORMAL
        self.radio_mode_dhuertas["state"] = NORMAL


class ModelFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Model")
        self._model = Models.SBOX
        self.var_model = StringVar(value=self._model)
        self.radio_model_sbox_r0 = Radiobutton(self,
                                               text="SBox round 0",
                                               variable=self.var_model,
                                               value=Models.SBOX,
                                               command=self.on_click)
        self.radio_model_sbox_r0.grid(row=0, column=0, sticky=W, padx=4)

        self.radio_model_inv_sbox_r10 = Radiobutton(self,
                                                    text="InvSBox round 10",
                                                    variable=self.var_model,
                                                    value=Models.INV_SBOX,
                                                    command=self.on_click)
        self.radio_model_inv_sbox_r10.grid(row=1, column=0, sticky=W, padx=4)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model is None:
            return
        self._model = model
        self.var_model.set(model)

    def on_click(self):
        self._model = int(self.var_model.get())

    def lock(self):
        self.radio_model_sbox_r0["state"] = DISABLED
        self.radio_model_inv_sbox_r10["state"] = DISABLED

    def unlock(self):
        self.radio_model_sbox_r0["state"] = NORMAL
        self.radio_model_inv_sbox_r10["state"] = NORMAL


class PlotFrame(LabelFrame):
    class Mode(Enum):
        STATISTICS = 0
        CORRELATION = 1

    def __init__(self, master):
        super().__init__(master, text="Plots")
        self._mode = PlotFrame.Mode.CORRELATION
        self._old_mode = PlotFrame.Mode.CORRELATION
        self._byte = None
        self._old_byte = None
        self.var_mode = IntVar(value=self._mode.value)
        self.var_byte = StringVar(value=0)

        self.label_byte = Label(self, text="Byte")
        self.label_byte.grid(row=0, column=0, sticky=W, padx=4)
        self.entry_byte = Entry(self, textvariable=self.var_byte)
        self.entry_byte.grid(row=0, column=1, sticky=EW, padx=4)

        self.radio_mode_stats = Radiobutton(self,
                                            text="Statistics",
                                            variable=self.var_mode,
                                            value=0)
        self.radio_mode_stats.grid(row=1, column=0, sticky=W, padx=4)

        self.radio_mode_corr = Radiobutton(self,
                                           text="Correlation",
                                           variable=self.var_mode,
                                           value=1)
        self.radio_mode_corr.grid(row=2, column=0, sticky=W, padx=4)

    def _validate_byte(self):
        byte = None
        try:
            byte = int(self.var_byte.get())
            valid = byte >= 0
        except ValueError:
            valid = False
        if valid:
            self._old_byte = self._byte
            self._byte = byte
        else:
            self._byte = None
        _set_validation_fg(self.label_byte, valid)
        return valid

    @property
    def changed(self):
        return ((self._old_byte or 0) != (self._byte or 0)) or (self._old_mode != self._mode)

    @property
    def mode(self):
        return self._mode

    @property
    def byte(self):
        return self._byte

    def validate(self):
        self.on_click()
        return self._validate_byte()

    def on_click(self):
        self._old_mode = PlotFrame.Mode(value=self._mode.value)
        self._mode = PlotFrame.Mode(value=self.var_mode.get())


class GeneralFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="General")
        Grid.columnconfigure(self, 1, weight=1)
        Grid.columnconfigure(self, 0, weight=1)

        self._iterations = None
        self._target = None
        self.var_iterations = StringVar()
        self.var_target = StringVar()

        self.label_iterations = Label(self, text="Iterations *")
        self.label_iterations.grid(row=0, column=0, sticky=W, padx=4)
        self.entry_iterations = Entry(self, textvariable=self.var_iterations)
        self.entry_iterations.grid(row=0, column=1, sticky=EW, padx=4)

        self.label_target = Label(self, text="Target *")
        self.label_target.grid(row=1, column=0, sticky=W, padx=4)
        self.entry_target = Entry(self, textvariable=self.var_target)
        self.entry_target.grid(row=1, column=1, sticky=EW, padx=4)

        self.frame_mode = ModeFrame(self)
        self.frame_mode.grid(row=2, column=0, sticky=NSEW, padx=4)

        self.frame_model = ModelFrame(self)
        self.frame_model.grid(row=2, column=1, sticky=NSEW, padx=4)

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations):
        if iterations is None:
            return
        self._iterations = iterations
        self.var_iterations.set(iterations)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        if target is None:
            return
        self._target = target
        self.var_target.set(target)

    def _validate_iterations(self):
        iterations = None
        try:
            iterations = max(int(self.var_iterations.get()), 0)
            valid = iterations > 0
        except ValueError:
            valid = False

        self._iterations = iterations if valid else None
        _set_validation_fg(self.label_iterations, valid)
        return valid

    def _validate_target(self):
        valid = False
        target = self.var_target.get()
        for port, *_ in serial.tools.list_ports.comports():
            if port == target:
                valid = True
                break

        self._target = target if valid else None
        _set_validation_fg(self.label_target, valid)
        return valid

    def validate(self):
        valid = self._validate_target()
        return self._validate_iterations() and valid

    def lock(self):
        self.entry_iterations["state"] = DISABLED
        self.entry_target["state"] = DISABLED
        self.frame_mode.lock()
        self.frame_model.lock()

    def unlock(self):
        self.entry_iterations["state"] = NORMAL
        self.entry_target["state"] = NORMAL
        self.frame_mode.unlock()
        self.frame_model.unlock()

    def clear(self):
        _clear_fg(self.label_iterations)
        _clear_fg(self.label_target)


class PerfsFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Performances")
        self._start = None
        self._end = None
        self._chunks = None
        self._verbose = False
        self._noise = False
        self.var_end = StringVar()
        self.var_start = StringVar()
        self.var_chunks = StringVar()
        self.var_verbose = BooleanVar(value=self._verbose)
        self.var_noise = BooleanVar(value=self._noise)

        self.label_start = Label(self, text="Start")
        self.label_start.grid(row=0, column=0, sticky=W, padx=4)
        self.entry_start = Entry(self, textvariable=self.var_start)
        self.entry_start.grid(row=0, column=1, padx=4)

        self.label_end = Label(self, text="End")
        self.label_end.grid(row=1, column=0, sticky=W, padx=4)
        self.entry_end = Entry(self, textvariable=self.var_end)
        self.entry_end.grid(row=1, column=1, sticky=W, padx=4)

        self.label_chunks = Label(self, text="Chunks")
        self.label_chunks.grid(row=2, column=0, sticky=W, padx=4)
        self.entry_chunks = Entry(self, textvariable=self.var_chunks)
        self.entry_chunks.grid(row=2, column=1, sticky=W, padx=4)

        self.label_verbose = Label(self, text="Verbose")
        self.label_verbose.grid(row=3, column=0, sticky=W, padx=4)
        self.check_verbose = Checkbutton(self,
                                         var=self.var_verbose,
                                         onvalue=True,
                                         offvalue=False,
                                         command=self.on_click)
        self.check_verbose.grid(row=3, column=1, padx=4)

        self.label_noise = Label(self, text="Noise")
        self.label_noise.grid(row=4, column=0, sticky=W, padx=4)
        self.check_noise = Checkbutton(self,
                                       var=self.var_noise,
                                       onvalue=True,
                                       offvalue=False,
                                       command=self.on_click)
        self.check_noise.grid(row=4, column=1, padx=4)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        if start is None:
            return
        self._start = start
        self.var_start.set(start)

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        if end is None:
            return
        self._end = end
        self.var_end.set(end)

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, chunks):
        if chunks is None:
            return
        self._chunks = chunks
        self.var_chunks.set(chunks)

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if verbose is None:
            return
        self._verbose = verbose
        self.var_verbose.set(verbose)

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, noise):
        if noise is None:
            return
        self._noise = noise
        self.var_noise.set(noise)

    def on_click(self):
        self._noise = self.var_noise.get()
        self._verbose = self.var_verbose.get()

    def _validate_start(self):
        start = self.var_start.get()
        try:
            start = int(start)
            valid = 0 <= start <= (self._end or start)
        except ValueError:
            valid = start == ""

        self._start = start if valid and start != "" else None
        _set_validation_fg(self.label_start, valid)
        return valid

    def _validate_end(self):
        end = self.var_end.get()
        try:
            end = int(end)
            valid = 0 <= (self._start or 0) <= end
        except ValueError:
            valid = end == ""

        self._end = end if valid and end != "" else None
        _set_validation_fg(self.label_end, valid)
        return valid

    def _validate_chunks(self):
        chunks = self.var_chunks.get()
        try:
            chunks = int(chunks)
            valid = chunks > 0
        except ValueError:
            valid = chunks == ""

        self._chunks = chunks if valid and chunks != "" else None
        _set_validation_fg(self.label_chunks, valid)
        return valid

    def validate(self):
        valid = self._validate_start()
        valid = self._validate_end() and valid
        return self._validate_chunks() and valid

    def lock(self):
        self.entry_start["state"] = DISABLED
        self.entry_end["state"] = DISABLED
        self.entry_chunks["state"] = DISABLED
        self.check_verbose["state"] = DISABLED
        self.check_noise["state"] = DISABLED

    def unlock(self):
        self.entry_start["state"] = NORMAL
        self.entry_end["state"] = NORMAL
        self.entry_chunks["state"] = NORMAL
        self.check_verbose["state"] = NORMAL
        self.check_noise["state"] = NORMAL

    def clear(self):
        _clear_fg(self.label_start)
        _clear_fg(self.label_end)
        _clear_fg(self.label_chunks)

    def update(self):
        _set_validation_fg(self.label_start, self._start)
        _set_validation_fg(self.label_end, self._end)
        _set_validation_fg(self.label_chunks, self._chunks)


class FilesFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Files")
        Grid.columnconfigure(self, 1, weight=1)
        self._path = os.path.abspath(os.sep.join(os.getcwd().split(os.sep)[:-3] + ["data"]))
        self.label_path = Label(self, text="Path *")
        self.label_path.grid(row=0, column=0, sticky=W, padx=4)
        self.var_path = StringVar(value=self._path)
        self.entry_path = Entry(self, textvariable=self.var_path)
        self.entry_path.grid(row=0, column=1, padx=4, sticky=EW)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if path is None:
            return
        self._path = path
        self.var_path.set(path)

    def _validate_path(self):
        path = self.var_path.get()
        valid = os.path.isdir(path) or os.path.isdir(os.sep.join(path.split(os.sep)[:-1]))
        self._path = path if valid else None
        _set_validation_fg(self.label_path, self._path)
        return valid

    def validate(self):
        return self._validate_path()

    def lock(self):
        self.entry_path["state"] = DISABLED

    def unlock(self):
        self.entry_path["state"] = NORMAL

    def clear(self):
        _clear_fg(self.label_path)

    def update(self):
        _set_validation_fg(self.label_path, self._path)


class ConfigFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Configuration")
        self.general = GeneralFrame(self)
        self.general.pack(side=TOP, expand=1, fill=BOTH)
        self._lower = Frame(self)
        self._lower.pack(side=TOP, expand=1, fill=BOTH)
        Grid.columnconfigure(self._lower, 0, weight=2)
        Grid.columnconfigure(self._lower, 1, weight=5)

        self.perfs = PerfsFrame(self._lower)
        self.perfs.grid(row=0, column=0, sticky=NSEW, padx=4)
        self.plot = PlotFrame(self._lower)
        self.plot.grid(row=0, column=1, sticky=NSEW, padx=4)
        self.file = FilesFrame(self)
        self.file.pack(side=TOP, expand=1, fill=BOTH)

    def validate(self):
        valid = self.general.validate()
        valid = self.perfs.validate() and valid
        valid = self.plot.validate() and valid
        return self.file.validate() and valid

    def lock(self):
        self.general.lock()
        self.perfs.lock()
        self.file.lock()

    def unlock(self):
        self.general.unlock()
        self.perfs.unlock()
        self.file.unlock()

    def clear(self):
        self.general.clear()
        self.perfs.clear()
        self.file.clear()

    def update(self):
        self.general.update()
        self.perfs.update()
        self.file.update()
