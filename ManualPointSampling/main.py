# -*- coding: UTF-8 -*-
from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
from mods import util
from mods.marker import Marker
from mods.canvasframe import Zoom_Advanced
from mods.dialoggridedit import DialogGridEdit

root= Tk()
root.title('point sampling')
root.geometry('1400x800')


frmMain = Frame(width=800, height=360)

frmMain.pack(side=LEFT, expand=True, fill=BOTH)
frmCanvas = Zoom_Advanced(frmMain)

hideImage = IntVar()
hideResult = IntVar()
hideScribble = IntVar()
showGrid = IntVar()
showGrid.set(True)
road_iterator_visualize_step = IntVar()
road_iterator_auto_init_seed = IntVar
road_iterator_auto_next = IntVar()
road_iterator_line_conn = IntVar()
road_iterator_line_conn.set(True)
road_iterator_auto_next.set(True)

def loadImg(fname = ''):
    if frmCanvas.history.canUndo() or frmCanvas.history.canRedo():
       if not messagebox.askokcancel("infomation","the image is changed, do you want to discard？"):
           return
    if fname == '':
        fname = filedialog.askopenfilename(title=u'open files', filetypes=[(u'all files','.*'), ("JPEG", ".jpg"), ("PNG", ".png"), ("BMP", ".bmp")])
    if fname == '':
        return
    frmCanvas.loadImage(fname)

def genPointSamples():
    if not showGrid.get():
        messagebox.showinfo('information','please show the grid first,because the patch size is presented in grid')
        return
    cnt = 0
    for o in frmCanvas.allscribble:
        if isinstance(o, Marker):
            cnt+=1
    if cnt==0:
        messagebox.showinfo('information','please marker the foreground or background first')
        return

    savedir = filedialog.askdirectory()
    frmCanvas.generate_points_training_samples(savedir)


def addHistory(img):
    frmCanvas.history.push(img)

def keypress( event ):
    print(event)
    thick = frmCanvas.thick
    if event.char=='+':
        thick += 1
    elif event.char=='-':
        thick -= 1

    if thick < 1:
        thick = 1
    frmCanvas.thick = thick

def toggleImg():
    frmCanvas.hideImage = hideImage.get()
    frmCanvas.show_image()
    pass

def toggleScribble():
    frmCanvas.hideScribble = hideScribble.get()
    frmCanvas.show_scribble()
    pass

def toggleResult():
    frmCanvas.hideResult = hideResult.get()
    frmCanvas.show_image()
    pass

def toggleGrid():
    frmCanvas.showGrid = showGrid.get()
    frmCanvas.show_grid()
    pass

def grid_edit():
    # grid_width = askinteger('input please','the width of the grid:',initialvalue=64)
    # grid_height = askinteger('input please','the height of the grid:',initialvalue=64)
    dialog = DialogGridEdit(frmMain, frmCanvas.gridinfo)
    frmMain.wait_window(dialog)
    if dialog.gridinfo is not None:
        frmCanvas.gridinfo = dialog.gridinfo
        showGrid.set(1)
        toggleGrid()

def toggleGrid():
    frmCanvas.showGrid = showGrid.get()
    frmCanvas.show_grid()
    pass

def toggleAutoNext():

    pass


def about():
    messagebox.showinfo('information','Author:Renbao Lian\n FuZhou University, Fuzhou, China\n Email: roblian@outlook.com \nAll rights reserved')
    pass

def memo():
    pass

def mahalanobis():
    trainpixels = frmCanvas.getFgScribbles()
    if trainpixels.__len__() == 0:
        messagebox.showinfo('information','please draw scribbles for objects')
        return

    mb = util.mahalanobis(frmCanvas.image, trainpixels)
    mb = (mb * 255).astype(np.uint8)
    frmCanvas.image = Image.fromarray(mb, mode="L")
    frmCanvas.show_image()
    addHistory(frmCanvas.image)

    pass

def euclidean():
    trainpixels = frmCanvas.getFgScribbles()
    if trainpixels.__len__() == 0:
        messagebox.showinfo('information','please draw scribbles for background')
        return

    mb = util.euclidean(frmCanvas.image, trainpixels)
    mb = (mb * 255).astype(np.uint8)
    frmCanvas.image = Image.fromarray(mb, mode="L")
    frmCanvas.show_image()
    addHistory(frmCanvas.image)

    pass

def reload():
    if len(frmCanvas.history)==0:
        return
    frmCanvas.image = frmCanvas.history.reload()
    frmCanvas.show_image()
def nothing(val):
    pass

def on_closing():
    if not frmCanvas.sampled:
        if not messagebox.askokcancel('question','Some markers have not sampled, do you really to exit?'):
                return

    root.destroy()

# create a top menu
menubar = Menu(root)

filemenu = Menu(menubar, tearoff=False)
filemenu.add_command(label="Load an Image", command=loadImg)
filemenu.add_separator()
filemenu.add_command(label="gen point samples", command=genPointSamples)
filemenu.add_separator()
filemenu.add_command(label="The Origin Image", command=reload)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

viewmenu = Menu(menubar, tearoff=False)
viewmenu.add_checkbutton(label="Hide the Scribbles", command=toggleScribble, variable=hideScribble)
viewmenu.add_checkbutton(label="Hide the Image", command=toggleImg, variable=hideImage)
# viewmenu.add_checkbutton(label="Hide the Result", command=toggleResult, variable=hideResult)
viewmenu.add_separator()
gridmenu = Menu(viewmenu, tearoff=False)
gridmenu.add_command(label='Edit', command=grid_edit)
gridmenu.add_checkbutton(label="Show Grid", command=toggleGrid, variable=showGrid)
viewmenu.add_cascade(label='Grid', menu=gridmenu)
menubar.add_cascade(label="View", menu=viewmenu)

helpmenu = Menu(menubar, tearoff=False)
helpmenu.add_command(label="About us...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)
root.config(menu=menubar)

root.bind( "<KeyPress>", keypress) #https://www.cnblogs.com/hhh5460/p/6701817.html
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
