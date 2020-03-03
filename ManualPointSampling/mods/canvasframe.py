# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
from mods.marker import Marker
from mods.util import *


class History(object):
    def __init__(self):
        self.images=[]
        self.index = -1

    def __len__(self):
        return len(self.images)

    def undo(self):
        if self.index > 0 :
            self.index-=1

        return self.images[self.index]

    def redo(self):
        if self.index < self.__len__()-1:
            self.index+=1

        return self.images[self.index]

    def reload(self):
        if self.index >= 0:
            self.index = 0;

        return self.images[self.index]

    def push(self, img):
        for i in range(self.index+1, self.__len__()):
            self.images.pop()

        self.images.append(img)
        self.index += 1

    def clear(self):
        self.images.clear()
        self.index = -1

    def canUndo(self):
        return self.index>0

    def canRedo(self):
        return self.index < self.__len__()-1


class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

class Zoom_Advanced(ttk.Frame):
    #private properties
    __index = 0
    __xx = None
    __yy = None
    __nowscribble = None
    __nowrect = None
    __fgcolor = 'red'
    __bgcolor = 'green'
    __rectwidth = 3
    __fgtype = 1
    __bgtype = 0
    __rectcolor='yellow'
    __recttype = 2
    __scribble_show_finish_index = 0
    __dragTag = None

    sampled = True

    #public properties
    hideImage = 0
    hideScribble = 0
    hideResult = 0
    hideMask = 0
    showGrid = 0
    gridinfo = {'offsetx':0, 'offsety':0,'width':64, 'height' : 64}
    thick = 3
    allscribble = []
    patch_size = 64
    imagename = None
    dirname = None
    seeds = [] # save seeds selected by user for deepwindow (iterative sliding window)
    ''' Advanced zoom of the image '''
    def __init__(self, mainframe, path=None, patch_size=64):
        self.patch_size = patch_size
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0, cursor='circle',
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        #add pop menus
        self.mouseMode = 0 #1999 drag, 1 stroke foreground 2 stroke background 3 draw rectangle 4 select seed for sliding window algorithm
        self.popmenu = tk.Menu(self, tearoff=False)
        self.popmenu.add_command(label='Drag All the Scene',command = self.setMouseMode0)
        self.popmenu.add_command(label='Mark Foreground by Points', command = self.setMouseMode5)
        self.popmenu.add_command(label='Mark Background by Points', command = self.setMouseMode6)
        self.popmenu.add_separator()
        self.popmenu.add_command(label='Origin Size',command = self.setScale1)
        self.popmenu.add_command(label='Zooming in 2 times',command = self.setScale2)
        self.popmenu.add_command(label='Zooming in 4 times',command = self.setScale4)
        self.popmenu.add_command(label='Zooming in 8 times',command = self.setScale8)
        self.popmenu.add_separator()
        self.popmenu.add_command(label='Zooming out 2 times',command = self.setScaleHalf)
        self.popmenu.add_command(label='Zooming out 4 times',command = self.setScaleQuater)



        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.lpress)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        self.canvas.bind('<Motion>',     self.mouse_move)
        self.canvas.bind('<ButtonRelease-3>', self.rrelease)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.canvas.bind("<ButtonRelease-1> ", self.lrelease)

        if path == None:
            self.image = None
            self.width, self.height = 300, 300
        else:
            self.image = Image.open(path)
            self.width, self.height = self.image.size

        self.result = None
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=1, outline='gray', tags='container')
        self.show_image()

        self.history = History()

    def undo(self):
        if self.history.canUndo():
            self.image = self.history.undo()
            self.show_image()

    def redo(self):
        if self.history.canRedo():
            self.image = self.history.redo()
            self.show_image()


    def mouse_move(self, event ):
        # check the object at the current cursor, change the cursor according to the object type
        item = self.canvas.gettags('current')
        # the point marker
        if len(item)>1 and 'point-' in item[0]:
            self.canvas.config(cursor='hand2')
        else:
            if self.mouseMode == 0 : # drag
                self.canvas.config(cursor='hand2')
            elif self.mouseMode == 5: # mark a foreground
                self.canvas.config(cursor='target')
            elif self.mouseMode == 6: # mark a foreground
                self.canvas.config(cursor='circle')



    def loadImage(self, path):
        if not self.sampled:
            if not messagebox.askokcancel('question','Some markers have not sampled, do you really to load another image?'):
                return
        [self.dirname,self.imagename]=os.path.split(path)

        self.image = Image.open(path)
        self.imscale = 1.0  # scale for the canvaas image
        self.width, self.height, = self.image.size

        self.history.clear()
        self.history.push(self.image)

        # Put image into container rectangle and use it to set proper coordinates to the image
        self.canvas.delete('container')
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=1, outline='red',tags='container')
        self.result = None
        self.eraseScribble()
        self.allscribble.clear()
        self.show_image()
        self.showGrid = True
        self.show_grid()
        self.sampled = True

    def rrelease(self, event):
        self.popmenu.post(event.x_root,event.y_root)
        self.__posx = event.x_root
        self.__posx = event.x_root
        self.__posy = event.y_root
        pass


    def setMouseMode0(self):
        self.mouseMode = 0
    def setMouseMode5(self):
        self.mouseMode = 5
    def setMouseMode6(self):
        self.mouseMode = 6

    def scaleX(self, scale):
        if not hasattr(self, '__posx'):
            bbox1 = self.canvas.bbox(self.container)
            self.__posx = bbox1[0]
            self.__posy = bbox1[1]

        scalex = self.canvas.canvasx(self.__posx)
        scaley = self.canvas.canvasy(self.__posy)
        self.canvas.scale('all', scalex, scaley, scale, scale)
        self.show_image()

    def setScale1(self):
        scale = 1
        scale /= self.imscale
        self.imscale = 1
        self.scaleX(scale)

    def setScaleHalf(self):
        scale = 0.5
        scale /= self.imscale
        self.imscale = 0.5
        self.scaleX(scale)

    def setScaleQuater(self):
        scale = 0.25
        scale /= self.imscale
        self.imscale = 0.25
        self.scaleX(scale)


    def setScale2(self):
        scale = 2
        scale /= self.imscale
        self.imscale = 2
        self.scaleX(scale)

    def setScale4(self):
        scale = 4
        scale /= self.imscale
        self.imscale = 4
        self.scaleX(scale)

    def setScale8(self):
        scale = 8
        scale /= self.imscale
        self.imscale = 8
        self.scaleX(scale)

    def realCoords(self, scalex, scaley):
        bbox1 = self.canvas.bbox(self.container)
        x , y = (scalex-bbox1[0] - 1) / self.imscale, (scaley-bbox1[1] - 1) / self.imscale
        x , y = int(x + 0.5), int(y + 0.5)

        return x, y


    def canvasCoords(self, x, y):
        bbox1 = self.canvas.bbox(self.container)
        scalex , scaley = (x * self.imscale + bbox1[0] + 1) , (y * self.imscale + bbox1[1] + 1)
        scalex , scaley = int(scalex + 0.5), int(scaley + 0.5)

        return scalex, scaley


    def lpress(self, event):
        if self.mouseMode == 0:
            ''' Remember previous coordinates for scrolling with the mouse '''
            self.canvas.scan_mark(event.x, event.y)
            return
        elif self.mouseMode == 5: # mark a foreground (do in lrelease)
            color = ''
            type = None
        elif self.mouseMode == 6: # mark a background (do in lrelease)
            color = ''
            type = None


        thick = self.thick * self.imscale

        self.__xx, self.__yy = self.canvas.canvasx( event.x - 1 ), self.canvas.canvasy( event.y - 1 )

        cx, cy = self.__xx, self.__yy
        rx, ry = self.realCoords(cx, cy)
        if rx < 0 or ry < 0 or rx > self.width or ry > self.height:
            return


    def removeScribbleByTags(self, tags):
        for scribble in self.allscribble:
            if scribble.tags == tags:
                self.canvas.delete(scribble.tags + str(scribble.index))
                self.allscribble.remove(scribble)
                if self.__scribble_show_finish_index>0:
                    self.__scribble_show_finish_index-=1
        # self.show_image()
        pass

    def removeScribbleByTag(self, tag):
        for scribble in self.allscribble:
            if (scribble.tags + str(scribble.index)) == tag:
                self.canvas.delete(tag)
                self.allscribble.remove(scribble)
                if self.__scribble_show_finish_index>0:
                    self.__scribble_show_finish_index-=1
                break
        pass

    def eraseScribble(self):
        for scribble in self.allscribble:
            self.canvas.delete(scribble.tags + str(scribble.index))

        self.__scribble_show_finish_index = 0
        pass

    def move_to(self, event):
        x, y = self.canvas.canvasx(event.x - 1 ), self.canvas.canvasy(event.y - 1 )
        if self.mouseMode == 0:
            ''' Drag (move) canvas to the new position '''
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.show_image()  # redraw the image
        elif self.mouseMode == 5 or self.mouseMode==6:
            # check the object at the current cursor, change the cursor according to the object type
            item = self.canvas.gettags('current')

            if len(item)>1 and 'point-' in item[0]:
                #move the object
                dx = x - self.__xx
                dy = y - self.__yy
                self.canvas.move(item[0], dx, dy)
                self.__dragTag = item[0]
        else:
            pass

        self.__xx = x
        self.__yy = y

    def lrelease(self, event):
        if self.mouseMode == 0:
            return
        elif self.mouseMode == 5 or self.mouseMode == 6: # mark a foreground or background
            if self.__dragTag is not None:
                self.removeScribbleByTag(self.__dragTag)
                self.__dragTag = None


            x, y = self.canvas.canvasx(event.x - 1 ), self.canvas.canvasy(event.y - 1 )
            rx, ry = self.realCoords(x, y)

            if rx < 0 or ry < 0 or rx > self.width or ry > self.height:
                return


            if self.mouseMode == 5:
                color = 'red'
                tags = 'point-fore'
            else:
                color = 'yellow'
                tags = 'point-back'
            point = [rx, ry]

            self.__index += 1
            mark_t = Marker(point=point, marker='o', color=color, thick=10, tags=tags, index=self.__index)
            self.show_mark(mark_t)

            nowobj = mark_t
            self.sampled = False

        self.allscribble.append(nowobj)

        pass

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()

    def show_grid(self):
        self.canvas.delete('grid_lines')
        if not self.showGrid:
            return

        bbox1 = self.canvas.bbox(self.container)  # get image area

        intval_w = int(self.gridinfo['width'] * self.imscale)
        intval_h = int(self.gridinfo['height'] * self.imscale)
        offset_x = int(self.gridinfo['offsetx'] * self.imscale) % intval_w
        offset_y = int(self.gridinfo['offsety'] * self.imscale) % intval_h

        bbox_left = bbox1[0]
        bbox_top = bbox1[1]
        bbox_width = bbox1[2]
        bbox_height = bbox1[3]
        for r in range(0, bbox_height, intval_h):
            if (r+bbox_top+offset_y)<bbox_height:
                self.canvas.create_line(bbox_left, r+bbox_top+offset_y, bbox_left+bbox_width, r+bbox_top+offset_y, fill='white', tags='grid_lines')

        for c in range(0, bbox_width, intval_w):
            if (c+bbox_left+offset_x)<bbox_width:
                self.canvas.create_line(c+bbox_left+offset_x, bbox_top, c+bbox_left+offset_x, bbox_height+bbox_top, fill='white', tags='grid_lines')

        pass


    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            self.canvas.delete('result')
            if not (self.hideResult==1 or self.result is None):
                result = self.result.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
                resulttk = ImageTk.PhotoImage(result.resize((int(x2 - x1), int(y2 - y1))))
                resultid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                                   anchor='nw', image=resulttk, tags='result')
                self.canvas.lower(resultid)  # set image into background
                self.canvas.resulttk = resulttk  # keep an extra reference to prevent garbage-collection

            self.canvas.delete('image')
            if not (self.hideImage == 1 or self.image is None):

                image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
                imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
                imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                                   anchor='nw', image=imagetk, tags='image')
                self.canvas.lower(imageid)  # set image into background
                self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection


    def clear_scribble(self):
        for scribble in self.allscribble:
            self.canvas.delete(scribble.tags + str(scribble.index))

    def show_scribble(self, erase=True):
        if erase:
            self.eraseScribble()

        if self.hideScribble == 1:
            return

        for index in range(self.__scribble_show_finish_index, len(self.allscribble)):
            scribble = self.allscribble[index]
            if isinstance(scribble, Marker):
                self.show_mark(scribble)

        self.__scribble_show_finish_index = len(self.allscribble) -1 if len(self.allscribble)>0 else 0



    def show_mark(self, scribble):
        if not isinstance(scribble, Marker):
            return

        x, y = scribble.point
        cx, cy = self.canvasCoords(x, y)
        thick = scribble.thick * self.imscale
        color = scribble.color
        marker = scribble.marker
        if marker == 'o':
            self.canvas.create_oval( cx - thick / 2, cy-thick / 2, cx+thick / 2, cy+thick / 2, width=0, fill = color, tags=scribble.tags + str(scribble.index))
        elif marker == 'x':
            self.canvas.create_line( cx - thick / 2, cy-thick / 2, cx+thick / 2, cy+thick / 2, width=thick / 2, fill = color, tags=scribble.tags + str(scribble.index))
            self.canvas.create_line( cx - thick / 2, cy+thick / 2, cx+thick / 2, cy-thick / 2, width=thick / 2, fill = color, tags=scribble.tags + str(scribble.index))
        elif marker == '+':
            self.canvas.create_line( cx - thick / 2, cy, cx+thick / 2, cy, width=thick / 2, fill = color, tags=scribble.tags + str(scribble.index))
            self.canvas.create_line( cx, cy-thick / 2, cx, cy+thick / 2, width=thick / 2, fill = color, tags=scribble.tags + str(scribble.index))
        elif marker == 's':
            self.canvas.create_rectangle(cx - thick / 2, cy-thick / 2, cx+thick / 2, cy+thick / 2, width=0, fill = color, tags=scribble.tags + str(scribble.index))

    def scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image


    def generate_points_training_samples(self, save_dir):
        np_img = np.asarray(self.image)
        tmp = self.allscribble.copy()
        while(True):
            if len(tmp)==0:
                break

            for o in self.allscribble:
                if o not in tmp:
                    continue
                if isinstance(o, Marker):
                    if 'point-' not in o.tags:
                        continue
                    [x, y] = o.point
                    left = (x-self.gridinfo['offsetx']) // self.gridinfo['width'] * self.gridinfo['width'] + self.gridinfo['offsetx']
                    top = (y-self.gridinfo['offsety']) // self.gridinfo['height'] * self.gridinfo['height'] + self.gridinfo['offsety']
                    width = self.gridinfo['width'] if left + self.gridinfo['width'] < self.width else self.width-left
                    height = self.gridinfo['height'] if top + self.gridinfo['height'] < self.height else self.height-top

                    if len(np_img.shape)==2:
                        img = np.zeros((self.gridinfo['height'], self.gridinfo['width']), dtype=np.uint8)
                    else:
                        img = np.zeros((self.gridinfo['height'], self.gridinfo['width'], np_img.shape[2]), dtype=np.uint8)

                    img[:height, :width, :] = np_img[top:top+height,left:left+width, :]
                    centers = []

                    if 'point-fore' in o.tags:
                        centers.append((y-top) * self.gridinfo['width'] + (x-left))
                        tmp.remove(o)
                        for oo in self.allscribble:
                            if oo==o:
                                continue
                            if not isinstance(o, Marker):
                                continue

                            [xx, yy] = oo.point
                            if xx > left and xx < left + self.gridinfo['width'] and yy > top and yy < top + self.gridinfo['height']:
                                centers.append((yy-top) * self.gridinfo['width'] + (xx-left))

                                tmp.remove(oo)
                    else:
                        tmp.remove(o)

                    gt = make_gt(img, centers, (self.gridinfo['width'], self.gridinfo['height']), 4.0)

                    if 'point-fore' in o.tags:
                        have_road = 1
                    else:
                        have_road = 0



                    imgid = os.path.splitext(self.imagename)[0]
                    gt = (gt * 255).astype(np.uint8)

                    img_gt = Image.fromarray(gt)
                    filename = os.path.join(save_dir, '%s_r%d_%04d_%04d_gt.png' % (imgid,have_road,left,top))
                    img_gt.save(filename, 'PNG')

                    np.save(os.path.join(save_dir , '%s_r%d_%04d_%04d_gt.npy' % (imgid,have_road,left,top)), gt)

                    img_patch = Image.fromarray(img)
                    img_patch.save(os.path.join(save_dir , '%s_r%d_%04d_%04d_img.png' % (imgid,have_road,left,top)), 'PNG')



        self.sampled = True
