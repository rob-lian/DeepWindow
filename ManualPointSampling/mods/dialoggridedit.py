# coding:utf8
import tkinter as tk
class DialogGridEdit(tk.Toplevel):
  def __init__(self, parent, grid):
    tk.Toplevel.__init__(self, parent)#super().__init__(self, parent)
    self.transient(parent)
    self.parent = parent
    self.gridinfo = grid
    self.title('grid set')
    # 弹窗界面
    self.setup_UI()

  def setup_UI(self):
    # 第0-0行（两列）
    row1 = tk.Frame(self)
    row1.pack(fill="x")
    tk.Label(row1, text='offset-x：', width=8).pack(side=tk.LEFT)
    self.__offsetx = tk.IntVar()
    self.__offsetx.set(self.gridinfo['offsetx'])
    tk.Entry(row1, textvariable=self.__offsetx, width=20).pack(side=tk.LEFT)
    # 第0-1行（两列）
    row1 = tk.Frame(self)
    row1.pack(fill="x")
    tk.Label(row1, text='offset-y：', width=8).pack(side=tk.LEFT)
    self.__offsety = tk.IntVar()
    self.__offsety.set(self.gridinfo['offsety'])
    tk.Entry(row1, textvariable=self.__offsety, width=20).pack(side=tk.LEFT)
    # 第一行（两列）
    row1 = tk.Frame(self)
    row1.pack(fill="x")
    tk.Label(row1, text='width：', width=8).pack(side=tk.LEFT)
    self.__width = tk.IntVar()
    self.__width.set(self.gridinfo['width'])
    tk.Entry(row1, textvariable=self.__width, width=20).pack(side=tk.LEFT)
    # 第二行
    row2 = tk.Frame(self)
    row2.pack(fill="x", ipadx=1, ipady=1)
    tk.Label(row2, text='height：', width=8).pack(side=tk.LEFT)
    self.__height = tk.IntVar()
    self.__height.set(self.gridinfo['height'])
    tk.Entry(row2, textvariable=self.__height, width=20).pack(side=tk.LEFT)
    # 第三行
    row3 = tk.Frame(self)
    row3.pack(fill="x")
    tk.Button(row3, text="   cancel   ", command=self.cancel).pack(side=tk.RIGHT)
    tk.Button(row3, text="   ok   ", command=self.ok).pack(side=tk.RIGHT)

    self.parent_width = self.parent.winfo_screenwidth()
    self.parent_height = self.parent.winfo_screenheight()
    width = 220
    height = 130
    x = self.parent_width / 2 - width / 2
    y = self.parent_height / 2 - height / 2
    self.geometry('%dx%d+%d+%d' % (width, height, x, y))

  def ok(self):
    self.gridinfo = {'offsetx':self.__offsetx.get(), 'offsety':self.__offsety.get(), 'width': self.__width.get(), 'height': self.__height.get()} # 设置数据
    self.destroy() # 销毁窗口
  def cancel(self):
    self.gridinfo = None # 空！
    self.destroy()
