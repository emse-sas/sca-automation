from tkinter import *


class LogFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Logging")
        self.text_log = Text(self, state=DISABLED)
        self.text_log.pack()
        self.text_log.tag_configure("last_insert", background="bisque")
        self.var_status = StringVar(value="Initialized")
        self.label_status = Label(self, textvariable=self.var_status)
        self.label_status.pack()

    def log(self, msg):
        self.text_log['state'] = NORMAL
        self.text_log.insert(INSERT, msg)
        self.text_log['state'] = DISABLED

    def insert_at_least(self, msg):
        self.text_log['state'] = NORMAL
        self.text_log.tag_remove("last_insert", "1.0", "end")
        self.text_log.insert("end", msg, "last_insert")
        self.text_log.see("end")
        self.text_log['state'] = DISABLED

    def overwrite_at_least(self, msg):
        self.text_log['state'] = NORMAL
        last_insert = self.text_log.tag_ranges("last_insert")
        self.text_log.delete(last_insert[0], last_insert[1])
        self.insert_at_least(msg)
        self.text_log['state'] = DISABLED

    def clear(self):
        self.text_log['state'] = NORMAL
        self.text_log.delete(1., END)
        self.text_log['state'] = DISABLED
