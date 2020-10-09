from tkinter import *


class LogFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Logging")
        self.text_log = Text(self, state=DISABLED)
        self.text_log.pack()
        self.var_status = StringVar(value="Initialized")
        self.label_status = Label(self, textvariable=self.var_status)
        self.label_status.pack()
