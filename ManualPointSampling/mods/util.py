# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import colorsys
import networkx as nx

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

def Normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    return (data-mn) / (mx - mn)

def make_gaussian(size, sigma=10, center=None):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        y0 = center[0]
        x0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
def make_gt(img, centers, outputRes=None, sigma=10):
    """ Make the ground-truth for each landmark.
    img: the original color image
    labels: the json labels with the Gaussian centers {'x': x, 'y': y}
    sigma: sigma of the Gaussian.
    """

    if outputRes is not None:
        h, w = outputRes
    else:
        h, w = img.shape
    # print (h, w, len(labels))
    #gt = np.zeros((h, w, len(labels)), np.float32)
    gt = np.zeros((h, w), np.float32)

    for land in centers:
        row = land // w
        col = land % w
        gt[:,:] = gt[:,:] + (make_gaussian((h, w), sigma, (row, col)))
    return gt

