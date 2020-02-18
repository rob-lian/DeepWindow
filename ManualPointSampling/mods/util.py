# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import colorsys

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

#矩阵归一化
def Normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    return (data-mn) / (mx - mn)

def euclidean(img, trainpixels):
    vals = []
    np_img = np.asarray(img)
    np_img = Normalize(np_img)
    rows, cols, channel = np_img.shape
    for tp in trainpixels:
        x,y = tp
        vals.append(np_img[y, x]) # y is rows and x is column

    vals = np.array(vals)
    mean = np.mean(vals, axis=0)
    res = np_img - mean #计算每个像素点的残差
    res = res.reshape(-1,3)
    au = np.sum(np.power(np.abs(res),0.5), axis=1)
    au = au.reshape(rows, cols)
    au = Normalize(au)
    au = 1 - au
    return au



def mahalanobis(img, trainpixels):
    vals = []
    np_img = np.asarray(img)
    rows, cols, channel = np_img.shape

    print(np_img.shape)
    for tp in trainpixels:
        x,y = tp
        vals.append(np_img[y, x]) # y is rows and x is column

    vals = np.array(vals)
    mean = np.mean(vals, axis=0)
    cov=np.cov(vals, rowvar=False) #每列为一个维度

    if np.linalg.norm(cov) == 0:#如果涂鸦是恒定颜色
        ones = np.ones(channel)
        inv = np.diag(ones)
    else:
        inv = np.linalg.inv(cov)
        inv = Normalize(inv)

    res = np_img - mean #计算每个像素点的残差
    # mahalabonis distance = (I(x) - mean).T * cov.inv * (I(x) - mean)
    res = res.reshape(-1,3)
    ma = np.dot(res, inv)
    ma = np.sum(ma * res, axis=1)
    ma = ma.reshape(rows, cols)

    ma = 1 - Normalize(ma)

    return ma
    pass


def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = []
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None

def getResult(mask):
    #根据mask返回半透明的遮罩
    tmp = np.zeros((mask.shape[0],mask.shape[1],4), dtype=np.uint8)
    tmp[mask==1] = (0,255,255,80)
    result = Image.fromarray(tmp, mode="RGBA")
    return result
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


if __name__ == '__main__':
    img = np.random.random((100,100,3))
    train = np.random.random((40,3))

    mean = np.mean(train, axis=0)
    cov = np.cov(train.T)
    inv = np.linalg.inv(cov)
    res = img - mean
    rows, cols, _ = res.shape
    for r in range(rows):
        for c in range(cols):
            p = res[r,c,:]
            p = np.expand_dims(p, axis=0)
            print(p.shape)
            ma = np.dot(p, inv)
            ma = np.dot(ma, p.T)
            print (ma)

