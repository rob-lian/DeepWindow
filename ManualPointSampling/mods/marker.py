# -*- coding: utf-8 -*-
class Marker():
    point=None
    marker = None
    color = None
    tag = None
    thick =None
    MakerType = ['o', 'x', '+', 's'] # s : squire
    index = 0

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

