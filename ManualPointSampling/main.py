# -*- coding: UTF-8 -*-
from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
from mods import util
from mods.marker import Marker
from mods.canvasframe import Zoom_Advanced
from mods.equalization import HistEqualRGB
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
        fname = filedialog.askopenfilename(title=u'open files', filetypes=[("JPEG", ".jpg"), ("PNG", ".png"), ("BMP", ".bmp"), (u'all files','.*')])
    if fname == '':
        return
    frmCanvas.loadImage(fname)
    checkMenuEnable()

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

def checkMenuEnable():
    if frmCanvas.history.canRedo():
        editmenu.entryconfigure('Redo', state=NORMAL)
    else:
        editmenu.entryconfigure('Redo', state=DISABLED)
    if frmCanvas.history.canUndo():
        editmenu.entryconfigure('Undo', state=NORMAL)
    else:
        editmenu.entryconfigure('Undo', state=DISABLED)


def addHistory(img):
    frmCanvas.history.push(img)
    checkMenuEnable()

def undo():
    frmCanvas.undo()
    checkMenuEnable()
    pass

def redo():
    frmCanvas.redo()
    checkMenuEnable()
    pass




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
    checkMenuEnable()

def histEqaulization():
    dst = np.asarray(frmCanvas.image)
    dst = HistEqualRGB(dst)
    frmCanvas.image = Image.fromarray(dst)
    # frmCanvas.image = util.HSVColor(frmCanvas.image)
    frmCanvas.show_image()
    addHistory(frmCanvas.image)
    pass

def nothing(val):
    pass



# create a top menu
menubar = Menu(root)

filemenu = Menu(menubar, tearoff=False)
filemenu.add_command(label="Load an Image", command=loadImg)
filemenu.add_separator()
filemenu.add_command(label="The Origin Image", command=reload)
filemenu.add_separator()
filemenu.add_command(label="gen point samples", command=genPointSamples)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

editmenu = Menu(menubar, tearoff=False)
editmenu.add_command(label="Undo", command=undo)
editmenu.add_command(label="Redo", command=redo)
menubar.add_cascade(label="Edit", menu=editmenu)

viewmenu = Menu(menubar, tearoff=False)
viewmenu.add_checkbutton(label="Hide the Scribbles", command=toggleScribble, variable=hideScribble)
viewmenu.add_checkbutton(label="Hide the Image", command=toggleImg, variable=hideImage)
viewmenu.add_checkbutton(label="Hide the Result", command=toggleResult, variable=hideResult)
viewmenu.add_separator()
gridmenu = Menu(viewmenu, tearoff=False)
gridmenu.add_command(label='Edit', command=grid_edit)
gridmenu.add_checkbutton(label="Show Grid", command=toggleGrid, variable=showGrid)
viewmenu.add_cascade(label='Grid', menu=gridmenu)
menubar.add_cascade(label="View", menu=viewmenu)

basemenu = Menu(menubar, tearoff=False)
colorcvtmenu = Menu(basemenu, tearoff = False)
colorcvtmenu.add_command(label = "Histogram Equalization", command=histEqaulization)
basemenu.add_cascade(label="Color Conversion", menu=colorcvtmenu)
menubar.add_cascade(label="Basic", menu=basemenu)

helpmenu = Menu(menubar, tearoff=False)
helpmenu.add_command(label="About us...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)
# 显示菜单
root.config(menu=menubar)

root.bind( "<KeyPress>", keypress) #https://www.cnblogs.com/hhh5460/p/6701817.html
root.mainloop()
