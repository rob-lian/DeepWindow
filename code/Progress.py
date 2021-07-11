#coding = utf-8
from tkinter import *
from tkinter import StringVar
import time


class Progress(Toplevel):
    """docstring for Progress"""

    def __init__(self, min, max):
        super().__init__()
        self.geometry('245x30')
        self.title('progress bar')
        self.wm_attributes('-topmost',1)

        row1 = Frame(self)
        row1.pack(fill="x")

        self.var = StringVar()
        self.var.set("start")
        self.label = Label(row1,textvariable = self.var,width = 5)
        self.label.grid(row = 0,column = 0,padx = 5)

        # craete rectangle with white background
        self.canvas = Canvas(row1,width = 245,height = 26,bg = "white")
        # creat a outline of rectangle(margin-left, margin-top, rect-width, rect-height), line-style, color
        self.out_line = self.canvas.create_rectangle(2,2,180,27,width = 1,outline = "black")
        self.canvas.grid(row = 0,column = 1,ipadx = 5)
        self.min = min
        self.max = max
        self.curr = min

    def step(self, step=1):
        fill_line = self.canvas.create_rectangle(2,2,0,27,width = 0,fill = "blue")
        self.curr += step
        percent = (self.curr - self.min) / (self.max-self.min)
        self.canvas.coords(fill_line, (0, 0, percent * 180, 30))
        self.var.set(str(round(percent * 100,1))+"%")
        if self.curr>=self.max:
            self.var.set(str("100%"))

        self.update()

if __name__ == '__main__':
    progress = Progress(0,100)
    for i in range(0,100):
        progress.step()
        time.sleep(0.1)
