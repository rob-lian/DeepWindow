#coding=utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
from skimage.morphology import dilation, erosion,disk

def Equalization(image):
    bg = dilation(erosion(image, disk(15)), disk(15))
    image2 = image - bg
    return image2

def HistEqualGray(image):
    image = image.astype(np.uint8)
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image2 = cv2.equalizeHist(image2)
    #gray to rgb
    image2 = np.array([image2, image2, image2])
    image2 = np.transpose(image2, (1,2,0))
    return image2

def HistEqualRGB(rgb):
    HSV = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    HSV_T = np.transpose(HSV, (2,0,1))
    V = HSV_T[2]
    V = cv2.equalizeHist(V)
    HSV_T[2] = V
    HSV = np.transpose(HSV_T, (1,2,0))
    rgb = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
    return rgb

def show_two_image(img, img_en):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_en)
    plt.show()

if __name__=='__main__':
    # filename = r'E:\study\data\DRIVE\test\images\19_test.tif'
    # filename = r'D:\EN.JPG'
    # img = Image.open(filename)
    # img = np.array(img)
    # img_en = HistEqualRGB(img)


    #随机从视网膜图像中裁剪一些小片，然后进行彩色增强处理，做对比观察

    db_root = r'E:\study\data\DRIVE'
    sv_root = r'E:\study\RoadExtraction\RoadExtractionByStrokGUI\roads\results\hist_eq_check'

    imglist = glob.glob(os.path.join(db_root,'training\images\*.tif'))

    if not os.path.exists(sv_root):
        os.makedirs(sv_root)

    # 按小片增强
    # for imgfile in imglist:
    #     print(imgfile)
    #     path,filename = os.path.split(imgfile)
    #     id, _ = os.path.splitext(filename)
    #     image = np.array(Image.open(imgfile))
    #     rows, cols = image.shape[0], image.shape[1]
    #     for i in range(10):
    #         r = random.randint(0,rows-32)
    #         c = random.randint(0,cols-32)
    #         patch = image[r:r+32, c:c+32]
    #         patch_en = HistEqualRGB(patch)
    #         fn = os.path.join(sv_root, id + '_' + str(i) + '_org.tif')
    #         cv2.imwrite(fn, patch)
    #         fn = os.path.join(sv_root, id + '_' + str(i) + '_en.tif')
    #         cv2.imwrite(fn, patch_en)
    #         show_two_image(patch, patch_en)
            # exit()

    # 按整体增强
    for imgfile in imglist:
        print(imgfile)
        path,filename = os.path.split(imgfile)
        id, _ = os.path.splitext(filename)
        maskfile = os.path.join(db_root, 'training\\mask\\' + id[:2] + '_training_mask.gif')
        mask = np.array(Image.open(maskfile))>0
        image = np.array(Image.open(imgfile))
        # image[~mask] = [0,0,0]
        # image_en = HistEqualRGB(image)
        # image_en = HistEqualGray(image)
        image_en = Equalization(image)
        show_two_image(image, image_en)

        # rows, cols = image.shape[0], image.shape[1]
        # for i in range(10):
        #     r = random.randint(0,rows-32)
        #     c = random.randint(0,cols-32)
        #     patch = image[r:r+32, c:c+32]
        #     patch_en = image_en[r:r+32, c:c+32]
        #     fn = os.path.join(sv_root, id + '_' + str(i) + '_org.tif')
        #     cv2.imwrite(fn, patch)
        #     fn = os.path.join(sv_root, id + '_' + str(i) + '_en.tif')
        #     cv2.imwrite(fn, patch_en)
        #     show_two_image(patch, patch_en)
        #     # exit()
