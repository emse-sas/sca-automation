from tkinter import *
from lib.cpa import Models
from lib.data import Request


class ModeFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Mode")
        self.var_mode = StringVar(value=Request.Modes.HARDWARE)
        self.radio_mode_hw = Radiobutton(self,
                                         text="Hardware",
                                         variable=self,
                                         value=Request.Modes.HARDWARE)
        self.radio_mode_hw.grid(row=0, column=0, sticky=W, padx=4)

        self.radio_mode_tiny = Radiobutton(self,
                                           text="Tiny",
                                           variable=self.var_mode,
                                           value=Request.Modes.TINY)
        self.radio_mode_tiny.grid(row=1, column=0, sticky=W, padx=4)

        self.radio_mode_ssl = Radiobutton(self,
                                          text="OpenSSL",
                                          variable=self.var_mode,
                                          value=Request.Modes.SSL)
        self.radio_mode_ssl.grid(row=2, column=0, sticky=W, padx=4)


class ModelFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Model")
        self.var_model = StringVar(value=Models.SBOX)
        self.radio_model_sbox_r0 = Radiobutton(self,
                                               text="SBox round 0",
                                               variable=self.var_model,
                                               value=Models.SBOX)
        self.radio_model_sbox_r0.grid(row=0, column=0, sticky=W, padx=4)

        self.radio_model_inv_sbox_r10 = Radiobutton(self,
                                                    text="InvSBox round 10",
                                                    variable=self.var_model,
                                                    value=Models.INV_SBOX)
        self.radio_model_inv_sbox_r10.grid(row=1, column=0, sticky=W, padx=4)


class GeneralFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Configuration")
        Grid.columnconfigure(self, 1, weight=1)
        Grid.columnconfigure(self, 0, weight=1)

        self.label_iterations = Label(self, text="Iterations")
        self.label_iterations.grid(row=0, column=0, sticky=W, padx=4)
        self.entry_iterations = Entry(self)
        self.entry_iterations.grid(row=0, column=1, sticky=EW, padx=4)

        self.label_target = Label(self, text="Target")
        self.label_target.grid(row=1, column=0, sticky=W, padx=4)
        self.entry_target = Entry(self)
        self.entry_target.grid(row=1, column=1, sticky=EW, padx=4)

        self.frame_mode = ModeFrame(self)
        self.frame_mode.grid(row=2, column=0, sticky=NSEW, padx=4)

        self.frame_model = ModelFrame(self)
        self.frame_model.grid(row=2, column=1, sticky=NSEW, padx=4)


class PerfsFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Performances")
        self.label_start = Label(self, text="Start")
        self.label_start.grid(row=0, column=0, sticky=W, padx=4)
        self.entry_start = Entry(self)
        self.entry_start.grid(row=0, column=1, padx=4)

        self.label_end = Label(self, text="End")
        self.label_end.grid(row=1, column=0, sticky=W, padx=4)
        self.entry_end = Entry(self)
        self.entry_end.grid(row=1, column=1, padx=4)

        self.label_chunks = Label(self, text="Chunks")
        self.label_chunks.grid(row=2, column=0, sticky=W, padx=4)
        self.entry_chunks = Entry(self)
        self.entry_chunks.grid(row=2, column=1, padx=4)

        self.var_verbose = BooleanVar(value=False)
        self.label_verbose = Label(self, text="Verbose")
        self.label_verbose.grid(row=3, column=0, sticky=W, padx=4)
        self.check_verbose = Checkbutton(self,
                                         var=self.var_verbose,
                                         onvalue=True,
                                         offvalue=False)
        self.check_verbose.grid(row=3, column=1, padx=4)

        self.var_noise = BooleanVar(value=False)
        self.label_noise = Label(self, text="Noise")
        self.label_noise.grid(row=4, column=0, sticky=W, padx=4)
        self.check_noise = Checkbutton(self,
                                       var=self.var_noise,
                                       onvalue=True,
                                       offvalue=False)
        self.check_noise.grid(row=4, column=1, padx=4)


class FilesFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Files")
        Grid.columnconfigure(self, 1, weight=1)
        self.label_path = Label(self, text="Path")
        self.label_path.grid(row=0, column=0, sticky=W, padx=4)
        self.entry_path = Entry(self)
        self.entry_path.grid(row=0, column=1, padx=4, sticky=EW)


class ConfigFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Configuration")
        self.general = GeneralFrame(self)
        self.general.pack(side=TOP, expand=1, fill=BOTH)
        self.perfs = PerfsFrame(self)
        self.perfs.pack(side=TOP, expand=1, fill=BOTH)
        self.file = FilesFrame(self)
        self.file.pack(side=TOP, expand=1, fill=BOTH)
