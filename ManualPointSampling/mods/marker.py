# -*- coding: utf-8 -*-
class Marker():
    point=None
    marker = None
    color = None
    tag = None #没有意义，只是给标记打一个标签
    thick =None
    MakerType = ['o', 'x', '+', 's'] # s : squire
    index = 0 #无实际意义，辅助绘图时撤销用

    def __init__(self, point, marker='o', color='yellow', thick=4, tags='start', index=0):
        if marker not in self.MakerType:
            raise RuntimeError('Marker type must be one of the :' + str(self.MakerType))
        self.point = point
        self.index = index
        self.marker = marker
        self.color = color
        self.tags = tags
        self.thick = thick
        pass

