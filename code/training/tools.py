import numpy as np
from PIL import Image
import cv2
import os
import random
import torch
from torch.utils.data import Dataset
import glob

def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name

class GrayHistEqual():
    # Gray the rgb image
    # Histogram equalization of gray image.

    def __init__(self):
        pass

    def __call__(self, sample):

        image, gt = sample['image'], sample['gt']

        image = image.astype(np.uint8)
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image2 = cv2.equalizeHist(image2)

        return {'image': image2,
                'gt': gt
                }

class RandomFlip():
    # Gray the rgb image
    # Histogram equalization of gray image.

    def __init__(self):
        pass

    def __call__(self, sample):

        image, gt = sample['image'], sample['gt']

        gt = np.squeeze(gt, axis=2)

        # random，0-do nothing，1-flip left right，2-flip up down，3-rotate 90 deg clockwise，4-rotate 180 deg clockwise，5-rotate 270 deg clockwise
        choice = random.randint(0,5)
        if choice==0:
            pass
        elif choice == 1:
            image = cv2.flip(image, 0)
            gt = cv2.flip(gt, 0)

        elif choice == 2:
            image = cv2.flip(image, 1)
            gt = cv2.flip(gt, 1)

        elif choice == 3:
            image = cv2.rotate(image, 0)
            gt = cv2.rotate(gt, 0)

        elif choice == 4:
            image = cv2.rotate(image, 1)
            gt = cv2.rotate(gt, 1)

        elif choice == 5:
            image = cv2.rotate(image, 2)
            gt = cv2.rotate(gt, 2)

        return {'image': image,
                'gt': gt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.astype(np.float32)
        gt = gt.astype(np.float32)

        if len(gt.shape) == 2:
            gt_tmp = gt
            h, w = gt_tmp.shape
            gt = np.zeros((h, w, 1), np.float32)
            gt[:,:,0] = gt_tmp
        if len(image.shape) == 2:
            image_tmp = image
            h, w = image_tmp.shape
            image = np.zeros((h, w, 3), np.float32)
            image[:,:,0] = image_tmp
            image[:,:,1] = image_tmp
            image[:,:,2] = image_tmp

        image = image.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt)
                }


class ToolDataset(Dataset):
    """Tool Dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,train=True,
                 inputRes=(64, 64),
                 outputRes=(64, 64),
                 sigma=5,
                 db_root_dir='',
                 transform=None,
                 save_vertices_indxs=False, dsname='mass'):
        self.train = train
        self.db_root_dir = db_root_dir
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.sigma = sigma
        self.transform = transform
        self.save_vertices_indxs = save_vertices_indxs
        self.dsname = dsname

        if dsname=='mass':
            if self.train:
                imgpath = db_root_dir + 'train/'
                clspath = db_root_dir + 'train/'
            else:
                imgpath = db_root_dir + 'val/'
                clspath = db_root_dir + 'val/'

        self.labels = glob.glob(os.path.join(imgpath, '*_gt.png'))
        self.images = glob.glob(os.path.join(clspath, '*_img.png'))

        self.images.sort()
        self.labels.sort()

        if len(self.images) != len(self.labels):
            raise RuntimeError('the training label set does not match the image set')

        for i in range(len(self.images)):
            _, imgfile = os.path.split(self.images[i])
            _, lblfile = os.path.split(self.labels[i])
            imgid = os.path.splitext(imgfile)[0][:-3]
            lblid = os.path.splitext(lblfile)[0][:-2]

            if imgid != lblid:
                raise RuntimeError('the lalbel file is not match its image in the {}th place, img: {}, lbl:{}'.format(i, imgfile, lblfile))

        print('Done initializing Dataset')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]), dtype=np.float32)
        gt = np.array(Image.open(self.labels[idx]), dtype=np.float32)
        gt = gt[:, :, np.newaxis]

        sample = {'image': img, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
