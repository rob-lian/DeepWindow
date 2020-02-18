from __future__ import division
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
import os
import random
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from skimage.morphology import skeletonize
import networkx as nx
import glob

import matplotlib.pyplot as plt

def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / (im.max() - im.min())
    return imn


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
    gt = np.zeros((h, w, 1), np.float32)

    for land in centers:
        row = land // w
        col = land % w
        gt[:,:,0] = gt[:,:,0] + (make_gaussian((h, w), sigma, (row, col)))
    return gt




def overlay_mask(img, mask, transparency=0.5):
    """
    Overlay a h x w x 3 mask to the image
    img: h x w x 3 image
    mask: h x w x 3 mask
    transparency: between 0 and 1
    """
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask[:, :, 0]) * img[:, :, 0] + mask[:, :, 0] * (
    255 * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask[:, :, 1]) * img[:, :, 1] + mask[:, :, 1] * (
    255 * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask[:, :, 2]) * img[:, :, 2] + mask[:, :, 2] * (
    255 * transparency + (1 - transparency) * img[:, :, 2])
    return im_over

def generate_graph_patch_undirect(pred):

    G=nx.Graph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx
            if pred[row_idx,col_idx] == 0: # 自己不是前景，不连接
                continue

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                if pred[row_idx,col_idx+1] == 1: # 前景才加入边
                    G.add_edge(node_idx,node_right_idx)

            if row_idx < pred.shape[0]-1 and col_idx > 0:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                if pred[row_idx+1,col_idx-1] == 1:
                    G.add_edge(node_idx,node_bottomleft_idx)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                if pred[row_idx+1,col_idx] == 1:
                    G.add_edge(node_idx,node_bottom_idx)

            if row_idx < pred.shape[0]-1 and col_idx < pred.shape[1]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                if pred[row_idx+1,col_idx+1] == 1:
                    G.add_edge(node_idx,node_bottomright_idx)

    return G


def generate_graph_patch(pred):

    G=nx.DiGraph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx

            if row_idx > 0 and col_idx > 0:
                node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
                cost = 1 - pred[row_idx-1,col_idx-1]
                G.add_edge(node_idx,node_topleft_idx,weight=cost)

            if row_idx > 0:
                node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
                cost = 1 - pred[row_idx-1,col_idx]
                G.add_edge(node_idx,node_top_idx,weight=cost)

            if row_idx > 0 and col_idx < pred.shape[1]-1:
                node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
                cost = 1 - pred[row_idx-1,col_idx+1]
                G.add_edge(node_idx,node_topright_idx,weight=cost)

            if col_idx > 0:
                node_left_idx = row_idx*pred.shape[1] + col_idx-1
                cost = 1 - pred[row_idx,col_idx-1]
                G.add_edge(node_idx,node_left_idx,weight=cost)

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                cost = 1 - pred[row_idx,col_idx+1]
                G.add_edge(node_idx,node_right_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx > 0:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                cost = 1 - pred[row_idx+1,col_idx-1]
                G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                cost = 1 - pred[row_idx+1,col_idx]
                G.add_edge(node_idx,node_bottom_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx < pred.shape[1]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                cost = 1 - pred[row_idx+1,col_idx+1]
                G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G


def find_output_connected_points(root_dir, save_vertices_indxs, train, img_idx, patch_size, img_filenames):

    if train:
        img_filename = img_filenames[img_idx]
        img = Image.open(os.path.join(root_dir, 'TrainSet', 'train_input', img_filename))
        mask_filename = img_filename[0:len(img_filename)-1]
        mask_gt = Image.open(os.path.join(root_dir, 'TrainSet', 'train_class', mask_filename))
    else:
        img_filename = img_filenames[img_idx]
        img = Image.open(os.path.join(root_dir, 'ValSet', 'val_input', img_filename))
        mask_filename = img_filename[0:len(img_filename)-1]
        mask_gt = Image.open(os.path.join(root_dir, 'ValSet', 'val_class', mask_filename))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    # 去掉图像中纯白的部分
    void_pixels = np.prod((img == np.array([255, 255, 255])), axis=2).astype(np.float32)
    void_pixels_eroded = ndimage.binary_erosion(void_pixels, structure=np.ones((5,5))).astype(void_pixels.dtype)
    void_pixels = ndimage.binary_dilation(void_pixels_eroded, structure=np.ones((5,5))).astype(void_pixels_eroded.dtype)
    valid_pixels = 1-void_pixels

    mask_gt = np.array(mask_gt)
    mask_gt_skeleton = skeletonize(mask_gt>0)

    mask_gt_skeleton_valid = mask_gt_skeleton*valid_pixels
    valid_indxs = np.argwhere(mask_gt_skeleton_valid==1)

    margin = int(np.round(patch_size/10.0))

    #Select a random point from the ground truth
    #print(img_idx)
    #print(img_filename)
    #print(valid_indxs)
    if len(valid_indxs) > 0:
        num_atempts = 0
        selected_point = random.randint(0,len(valid_indxs)-1)
        center = (valid_indxs[selected_point,1], valid_indxs[selected_point,0])

        # 保证切片在图片范围内
        while (center[0] < patch_size/2 or center[1] < patch_size/2 or center[0] >  w - patch_size/2 or center[1] >  h - patch_size/2) and num_atempts < 20:
            selected_point = random.randint(0,len(valid_indxs)-1)
            center = (valid_indxs[selected_point,1], valid_indxs[selected_point,0])
            num_atempts += 1

        if num_atempts < 20:

            #Add selected vertex to file to reproduce the experiments
            if save_vertices_indxs:
                f = open(os.path.join(root_dir, 'points_selected.txt'), 'a')
                f.write(str(img_idx) + " " + str(valid_indxs[selected_point,0]) + " " + str(valid_indxs[selected_point,1]) + "\n")
                f.close()

            x_tmp = int(center[0]-patch_size/2)
            y_tmp = int(center[1]-patch_size/2)
            img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

            #Find intersection points between skeleton and inner bbox from patch
            mask_gt_crop = mask_gt_skeleton_valid[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

            bbox_mask = np.zeros((patch_size,patch_size))
            bbox_mask[margin,margin:patch_size-margin] = 1
            bbox_mask[margin:patch_size-margin,margin] = 1
            bbox_mask[margin:patch_size-margin,patch_size-margin] = 1
            bbox_mask[patch_size-margin,margin:patch_size-margin] = 1

            intersection_bbox_with_gt = mask_gt_crop*bbox_mask

            #Discard intersection points not connected to the patch center
            G = generate_graph_patch(mask_gt_crop)
            idx_end = (patch_size/2)*patch_size + patch_size/2
            intersection_idxs = np.argwhere(intersection_bbox_with_gt==1)
            connected_intersection_idxs = []
            for ii in range(0,len(intersection_idxs)):
                idx_start = intersection_idxs[ii,0]*patch_size + intersection_idxs[ii,1]
                length_pred, path_pred = nx.bidirectional_dijkstra(G, idx_start, idx_end, weight='weight')
                if length_pred == 0:
                    connected_intersection_idxs.append(intersection_idxs[ii])

            output_points = np.asarray(connected_intersection_idxs)
            for ii in range(0,len(output_points)):
                tmp_value = output_points[ii,0]
                output_points[ii,0] = output_points[ii,1]
                output_points[ii,1] = tmp_value

            return img_crop, output_points

        else:

            output_points = []
            img_crop = []
            return img_crop, output_points

    else:
        output_points = []
        img_crop = []
        return img_crop, output_points

def find_output_connected_points_selected_point(root_dir, selected_vertex, train, img_idx, patch_size, center, img_filenames):

    if train:
        img_filename = img_filenames[img_idx]
        img = Image.open(os.path.join(root_dir, 'TrainSet', 'train_input', img_filename))
        mask_filename = img_filename[0:len(img_filename)-1]
        mask_gt = Image.open(os.path.join(root_dir, 'TrainSet', 'train_class', mask_filename))
    else:
        img_filename = img_filenames[img_idx]
        img = Image.open(os.path.join(root_dir, 'ValSet', 'val_input', img_filename))
        mask_filename = img_filename[0:len(img_filename)-1]
        mask_gt = Image.open(os.path.join(root_dir, 'ValSet', 'val_class', mask_filename))


    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    void_pixels = np.prod((img == np.array([255, 255, 255])), axis=2).astype(np.float32)
    void_pixels_eroded = ndimage.binary_erosion(void_pixels, structure=np.ones((5,5))).astype(void_pixels.dtype)
    void_pixels = ndimage.binary_dilation(void_pixels_eroded, structure=np.ones((5,5))).astype(void_pixels_eroded.dtype)
    valid_pixels = 1-void_pixels

    mask_gt = np.array(mask_gt)
    mask_gt_skeleton = skeletonize(mask_gt>0)

    mask_gt_skeleton_valid = mask_gt_skeleton*valid_pixels
    valid_indxs = np.argwhere(mask_gt_skeleton_valid==1)

    margin = int(np.round(patch_size/10.0))

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

    #Find intersection points between skeleton and inner bbox from patch
    mask_gt_crop = mask_gt_skeleton_valid[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    bbox_mask = np.zeros((patch_size,patch_size))
    bbox_mask[margin,margin:patch_size-margin] = 1
    bbox_mask[margin:patch_size-margin,margin] = 1
    bbox_mask[margin:patch_size-margin,patch_size-margin] = 1
    bbox_mask[patch_size-margin,margin:patch_size-margin] = 1

    intersection_bbox_with_gt = mask_gt_crop*bbox_mask

    #Discard intersection points not connected to the patch center
    G = generate_graph_patch(mask_gt_crop)
    idx_end = (patch_size/2)*patch_size + patch_size/2
    intersection_idxs = np.argwhere(intersection_bbox_with_gt==1)
    connected_intersection_idxs = []
    for ii in range(0,len(intersection_idxs)):
        idx_start = intersection_idxs[ii,0]*patch_size + intersection_idxs[ii,1]
        length_pred, path_pred = nx.bidirectional_dijkstra(G, idx_start, idx_end, weight='weight')
        if length_pred == 0:
            connected_intersection_idxs.append(intersection_idxs[ii])

    output_points = np.asarray(connected_intersection_idxs)
    for ii in range(0,len(output_points)):
        tmp_value = output_points[ii,0]
        output_points[ii,0] = output_points[ii,1]
        output_points[ii,1] = tmp_value

    return img_crop, output_points



class ToolDataset(Dataset):
    """Tool Dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,train=True,
                 inputRes=(64, 64),
                 outputRes=(64, 64),
                 sigma=5,
                 db_root_dir='',
                 transform=None,
                 save_vertices_indxs=False, online=True, have_road=True, dsname='mass'):
        """Loads image to label pairs for tool pose estimation
        db_elements: the names of the video files
        db_root_dir: dataset directory with subfolders "frames" and "Annotations"
        online: if true generate samples in realtime
        train: if true generate training samples, false to generate validation samples in realtime disregard parameter 'online'
        have_road: True means the training sample are all including roads
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.sigma = sigma
        self.transform = transform
        self.save_vertices_indxs = save_vertices_indxs
        self.online = online
        self.dsname = dsname

        if dsname=='mass':
            if self.train:
                imgpath = db_root_dir + 'TrainSet/train_input/'
                clspath = db_root_dir + 'TrainSet/train_class/'
            else:
                imgpath = db_root_dir + 'ValSet/val_input/'
                clspath = db_root_dir + 'ValSet/val_class/'
        elif dsname=='deepglobe':
            if self.train:
                imgpath = db_root_dir + 'train/'
                clspath = db_root_dir + 'train/'
            else:
                imgpath = db_root_dir + 'valid2/'
                clspath = db_root_dir + 'valid2/'
        elif dsname=='DRIVE':
            if self.train:
                imgpath = db_root_dir + 'training/images/'
                clspath = db_root_dir + 'training/1st_manual/'
            else:
                imgpath = db_root_dir + 'valid/images/'
                clspath = db_root_dir + 'valid/1st_manual/'

        if not online: # guided to the cropped patches
            if have_road:
                self.labels = glob.glob(os.path.join(db_root_dir, '*_r1_*_gt.png'))
                self.images = glob.glob(os.path.join(db_root_dir, '*_r1_*_img.png'))
            else:
                self.labels = glob.glob(os.path.join(db_root_dir, '*_gt.png'))
                self.images = glob.glob(os.path.join(db_root_dir, '*_img.png'))
        elif dsname=='mass':
            self.images = glob.glob(os.path.join(imgpath, '*.tiff'))
            self.labels = glob.glob(os.path.join(clspath, '*.tif'))
        elif dsname=='deepglobe':
            self.images = glob.glob(os.path.join(imgpath, '*.jpg'))
            self.labels = glob.glob(os.path.join(clspath, '*.png'))
        elif dsname=='DRIVE':
            self.images = glob.glob(os.path.join(imgpath, '*.tif'))
            self.labels = glob.glob(os.path.join(clspath, '*.gif'))

        self.images.sort()
        self.labels.sort()

        if len(self.images) != len(self.labels):
            raise RuntimeError('the training label set does not match the image set')

        for i in range(len(self.images)):
            _, imgfile = os.path.split(self.images[i])
            _, lblfile = os.path.split(self.labels[i])
            if not online:
                imgid = os.path.splitext(imgfile)[0][:-3]
                lblid = os.path.splitext(lblfile)[0][:-2]
            else:
                if dsname=='mass':
                    imgid = os.path.splitext(imgfile)[0]
                    lblid = os.path.splitext(lblfile)[0]
                elif dsname=='deepglobe':
                    imgid = os.path.splitext(imgfile)[0][:-3]
                    lblid = os.path.splitext(lblfile)[0][:-4]
                elif dsname=='DRIVE':
                    imgid = imgfile[0:imgfile.find('_')]
                    lblid = imgfile[0:imgfile.find('_')]
                else:
                    raise RuntimeError('dataset is not exist')

            if imgid != lblid:
                raise RuntimeError('the lalbel file is not match its image in the {}th place, img: {}, lbl:{}'.format(i, imgfile, lblfile))

        print('Done initializing Dataset')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.online:
            img, gt, mask_gt, have_road, crop_coords = self.make_img_gt_pair(idx)
        else:

            img = np.array(Image.open(self.images[idx]), dtype=np.float32)
            gt = np.array(Image.open(self.labels[idx]), dtype=np.float32)
            gt = gt[:,:, np.newaxis]
            mask_gt = []
            have_road = []
            crop_coords = []

        sample = {'image': img, 'gt': gt,
                  'mask_gt': mask_gt,
                  'have_road': have_road,
                  'crop_coords' : crop_coords}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        imagefile = self.images[idx]
        labelfile = self.labels[idx]
        patch_size = self.inputRes[0]
        sigma = self.sigma

        img = Image.open(imagefile)
        lbl = Image.open(labelfile)

        img = np.array(img, dtype=np.float32)
        h, w = img.shape[:2]

        if self.dsname=='mass':
            # 去掉图像中纯白的部分
            void_pixels = np.prod((img == np.array([255, 255, 255])), axis=2).astype(np.float32)
            void_pixels_eroded = ndimage.binary_erosion(void_pixels, structure=np.ones((5,5))).astype(void_pixels.dtype)
            void_pixels = ndimage.binary_dilation(void_pixels_eroded, structure=np.ones((5,5))).astype(void_pixels_eroded.dtype)
            valid_pixels = 1-void_pixels

        elif self.dsname =='deepglobe':
            # there is not invalid area in deepglobe
            valid_pixels = np.ones((h, w)).astype(np.float32)

        elif self.dsname =='DRIVE':
            # 去掉图像中纯黑的部分
            void_pixels = np.prod((img == np.array([0, 0, 0])), axis=2).astype(np.float32)
            void_pixels_eroded = ndimage.binary_erosion(void_pixels, structure=np.ones((5,5))).astype(void_pixels.dtype)
            void_pixels = ndimage.binary_dilation(void_pixels_eroded, structure=np.ones((5,5))).astype(void_pixels_eroded.dtype)
            valid_pixels = 1-void_pixels
        else:
            raise RuntimeError('dataset name is invalid')

        mask_gt = np.array(lbl)
        if len(mask_gt.shape) > 2:
            mask_gt = np.squeeze(mask_gt[:,:,1])

        # print(imagefile)
        # print(mask_gt.shape)
        # print(np.max(mask_gt))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(mask_gt, cmap='gray')
        # plt.show()
        # exit()

        mask_gt_skeleton = skeletonize(mask_gt > 0)
        mask_gt_skeleton_valid = mask_gt_skeleton * valid_pixels
        # print(set(mask_gt_skeleton.flat))

        valid_indxs = np.argwhere(mask_gt_skeleton_valid == 1)
        valid_pixels_indxs = np.argwhere(valid_pixels == 1)

        have_road = False
        if random.random() < 0.5 and len(valid_indxs) > 0: # sample from regions have roads

            selected_point = random.randint(0,len(valid_indxs)-1)
            center = (valid_indxs[selected_point,1], valid_indxs[selected_point,0])

            # 保证切片在图片范围内
            # del: because there are roads beside the image border when we iterate the test images
            # num_atempts = 0
            # while (center[0] < patch_size/2 or center[1] < patch_size/2 or center[0] >  w - patch_size/2 or center[1] >  h - patch_size/2) and num_atempts < 20:
            #     selected_point = random.randint(0,len(valid_indxs)-1)
            #     center = (valid_indxs[selected_point,1], valid_indxs[selected_point,0])
            #     num_atempts += 1

            # if num_atempts >= 20:
            #     print(imagefile + ' 找不到合适的切片. ')
            #     img_crop = np.zeros((patch_size,patch_size,3), dtype=np.float32)
            #     gt = np.zeros((patch_size,patch_size,1), dtype=np.float32)
            #     mask_gt = np.zeros((patch_size,patch_size,1), dtype=np.float32)
            #     return img_crop, gt, mask_gt

            # the center could not be fixed in the patch center, this will cause bias in the training
            y_tmp = int(center[1]-patch_size // 2)
            x_tmp = int(center[0]-patch_size // 2)

            # random perturbation the center position
            y_tmp += int((random.random() - 0.5) * patch_size / 2)
            x_tmp += int((random.random() - 0.5) * patch_size / 2)

        elif len(valid_pixels_indxs) > 0: #sample from valid pixels, not ensure have roads
            selected_point = random.randint(0, len(valid_pixels_indxs) - 1)
            center = (valid_pixels_indxs[selected_point, 1], valid_pixels_indxs[selected_point, 0])
            y_tmp = int(center[1] - patch_size // 2)
            x_tmp = int(center[0] - patch_size // 2)

        else:
            print(imagefile + ' 训练图无效')
            img_crop = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
            gt = np.zeros((patch_size, patch_size, 1), dtype=np.float32)
            mask_gt = np.zeros((patch_size, patch_size, 1), dtype=np.float32)
            return img_crop, gt, mask_gt, have_road, (0,0)

        # pad 64 with zeros , this is filiciate to crop patch
        mask_gt_skeleton_valid = np.pad(mask_gt_skeleton_valid, patch_size, 'constant')

        img = img.transpose((2, 0, 1))
        r = np.pad(img[0], patch_size, 'constant')
        g = np.pad(img[1], patch_size, 'constant')
        b = np.pad(img[2], patch_size, 'constant')
        img = np.dstack((r,g,b)) # 1500 *1500 * 3 !!!


        # adjust the y_tmp and x_tmp because pad 64 zeros on all sides
        y_tmp += patch_size
        x_tmp += patch_size
        mask_gt_crop = mask_gt_skeleton_valid[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]
        img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

        G = generate_graph_patch_undirect(mask_gt_crop)
        # num_components = nx.number_connected_components(G)
        subGs = [G.subgraph(c) for c in nx.connected_components(G) ]
        centers = [nx.center(c)[0] for c in subGs] # nx.center可能会返回2个值，当该网络偶数个节点。
        if len(centers) > 0:
            have_road = True

        gt = make_gt(img_crop, centers, (patch_size, patch_size), sigma)

        mask_gt_crop = mask_gt_crop[:, :, np.newaxis]

        # print(img_crop.shape)
        # print(gt.shape)
        # print(mask_gt_crop.shape)

        crop_coords = (y_tmp - patch_size, x_tmp - patch_size)
        return img_crop, gt, mask_gt_crop, have_road, crop_coords

    def store_gt_asmatfile(self):
        gt_tool = np.zeros((2, 3, len(self.img_list)), dtype=np.float32)
        for i in range(0, len(self.img_list)):
            temp = txt2mat(self.img_list[i])
            if temp.shape[0] == 0:
                gt_tool[:, :, i] = np.nan
            else:
                gt_tool[:, :, i] = np.transpose(temp)

        a = {'gt_tool': gt_tool}
        sio.savemat('gt_tool', a)

    def get_img_size(self):

        return self.inputRes


class RandomFlip():
    # Gray the rgb image
    # Histogram equalization of gray image.

    def __init__(self):
        pass

    def __call__(self, sample):

        image, gt, mask_gt, have_road, crop_coords = \
            sample['image'], sample['gt'], \
            sample['mask_gt'], sample['have_road'], sample['crop_coords']

        gt = np.squeeze(gt, axis=2)
        mask_gt = np.squeeze(mask_gt, axis=2)

        # 随机数，0-不动，1-左右翻转，2-上下翻转，3-顺时针90，4-顺时针180，5-顺时针270
        choice = random.randint(0,5)
        if choice==0:
            pass
        elif choice == 1:
            image = cv2.flip(image, 0) # 左右翻转
            gt = cv2.flip(gt, 0) # 左右翻转
            mask_gt = cv2.flip(mask_gt, 0) # 左右翻转
        elif choice == 2:
            image = cv2.flip(image, 1) # 上下翻转
            gt = cv2.flip(gt, 1) # 上下翻转
            mask_gt = cv2.flip(mask_gt, 1) # 上下翻转
        elif choice == 3:
            image = cv2.rotate(image, 0) # 旋转90°
            gt = cv2.rotate(gt, 0) # 旋转90°
            mask_gt = cv2.rotate(mask_gt, 0) # 旋转90°
        elif choice == 4:
            image = cv2.rotate(image, 1) # 旋转180°
            gt = cv2.rotate(gt, 1) # 旋转180°
            mask_gt = cv2.rotate(mask_gt, 1) # 旋转180°
        elif choice == 5:
            image = cv2.rotate(image, 2) # 旋转270°
            gt = cv2.rotate(gt, 2) # 旋转270°
            mask_gt = cv2.rotate(mask_gt, 2) # 旋转270°

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(image)
        # plt.subplot(122)
        # plt.imshow(image2)
        # plt.show()
        # exit()

        return {'image': image,
                'gt': gt,
                'mask_gt': mask_gt,
                'have_road': have_road,
                'crop_coords': crop_coords}


class GrayHistEqual():
    # Gray the rgb image
    # Histogram equalization of gray image.

    def __init__(self):
        pass

    def __call__(self, sample):

        image, gt, mask_gt, have_road, crop_coords = \
            sample['image'], sample['gt'], \
            sample['mask_gt'], sample['have_road'], sample['crop_coords']

        image = image.astype(np.uint8)
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image2 = cv2.equalizeHist(image2)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(image)
        # plt.subplot(122)
        # plt.imshow(image2)
        # plt.show()
        # exit()

        return {'image': image2,
                'gt': gt,
                'mask_gt': mask_gt,
                'have_road': have_road,
                'crop_coords': crop_coords}

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        maxRot (float): maximum rotation angle to be added
        maxScale (float): maximum scale to be added
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2

        sc = (self.scales[1] - self.scales[0]) * random.random() - \
             (self.scales[1] - self.scales[0]) / 2 + 1

        img, gt, mask_gt, have_road, crop_coords = \
            sample['image'], sample['gt'], \
            sample['mask_gt'], sample['have_road'], sample['crop_coords']

        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, sc)
        img_ = cv2.warpAffine(img, M, (w, h))

        h_gt, w_gt = gt.shape[:2]
        center_gt = (w_gt / 2, h_gt / 2)
        M = cv2.getRotationMatrix2D(center_gt, rot, sc)
        gt_ = cv2.warpAffine(gt, M, (w_gt, h_gt))

        return {'image': img_,
                'gt': gt,
                'mask_gt': mask_gt,
                'have_road': have_road,
                'crop_coords': crop_coords}


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        image, gt, valid_img = sample['image'], sample['gt'], sample['valid_img']

        if random.random() < 0.5:
            image = cv2.flip(image, flipCode=1)
            gt = cv2.flip(gt, flipCode=1)

        sample['image'], sample['gt'], sample['valid_img'] = image, gt, valid_img

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt, mask_gt, have_road, crop_coords = \
            sample['image'], sample['gt'], \
            sample['mask_gt'], sample['have_road'], sample['crop_coords']


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
                'gt': torch.from_numpy(gt),
                'mask_gt': mask_gt,
                'have_road': have_road,
                'crop_coords': crop_coords}


class normalize(object):

    def __init__(self, mean=[171.0773/255, 98.4333/255, 58.8811/255], std=[1.0, 1.0, 1.0]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        image, gt, valid_img = sample['image'], sample['gt'], sample['valid_img']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        sample['image'], sample['gt'], sample['valid_img'] = image, gt, valid_img

        return sample


if __name__ == '__main__':
    from torchvision.utils import make_grid
    import torch
    save_path_root = '../results/gt_train_check/'
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    a = ToolDataset(train=True)
    for i, sample in enumerate(a):
        img, gt, mask_gt = sample['image'], sample['gt'], sample['mask_gt']
        img = img / 255.0
        gt = np.squeeze(gt)
        mask_gt = np.squeeze(mask_gt)
        images = np.zeros((2, img.shape[0], img.shape[1], 3), dtype=np.float32)
        images[0,:,:,:] = img
        images[1,:,:,0] = gt
        images[1,:,:,1] = mask_gt
        images[1,:,:,2] = mask_gt
        images = (images * 255).astype(np.uint8)
        images = images.transpose([0, 3, 1, 2])
        images = torch.from_numpy(images)

        grid_image = make_grid(images, 2, normalize=False, range=(0, 255)).numpy().transpose([1,2,0])
        grid_image = Image.fromarray(grid_image)
        filename = os.path.join(save_path_root,'%04d.tif'  % i)
        grid_image.save(filename)
        print(filename + " saved")
        # print(grid_image)
        # plt.figure()
        # plt.imshow(grid_image)
        # plt.show(block=True)
        # break
