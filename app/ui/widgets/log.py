from tkinter import *


class LogFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Logging")
        self.text_log = Text(self, state=DISABLED)
        self.text_log.pack()
        self.var_status = StringVar(value="Initialized")
        self.label_status = Label(self, textvariable=self.var_status)
        self.label_status.pack()

    def log(self, msg, ):
        self.text_log['state'] = NORMAL
        self.text_log.insert(INSERT, msg)
        self.text_log['state'] = DISABLED

    def clear(self):
        self.text_log['state'] = NORMAL
        self.text_log.delete(1., END)
        self.text_log['state'] = DISABLED
