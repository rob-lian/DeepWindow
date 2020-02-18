# Includes
import os

import numpy as np
import scipy.misc
import torch
from PIL import Image
from torch.autograd import Variable
import json
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks

import roads.HourGlass as nt
import roads.UNet as un
import roads.patch.bifurcations_toolbox_roads as tb
import glob

# Setting of parameters
model='Hourglass'
epoch = 0

# mass
model_dir = '/roads/results'
db_root_dir = '/data/Mass_Point_Supervision/val_gt'
output_dir = model_dir + '/val_results/mass/' + model + '/' + str(epoch) + '/'
modelName = '/roads/results/weights/Hourglass_epoch-599.pth'


patch_size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters in p are used for the name of the model
def BuildNet(model):
    if torch.cuda.is_available():
        gpu_id = 0
    else:
        gpu_id = -1

    if model=='Hourglass':
        # Parameters in p are used for the name of the model
        p = {}
        p['useRandom'] = 1  # Shuffle Images
        p['useAug'] = 0  # Use Random rotations in [-30, 30] and scaling in [.75, 1.25]
        p['inputRes'] = (64, 64)  # Input Resolution
        p['outputRes'] = (64, 64)  # Output Resolution (same as input)
        p['g_size'] = 32  # Higher means narrower Gaussian
        p['trainBatch'] = 64  # Number of Images in each mini-batch
        p['numHG'] = 2  # Number of Stacked Hourglasses
        p['Block'] = 'ConvBlock'  # Select: 'ConvBlock', 'BasicBlock', 'BottleNeck', 'BottleneckPreact'

        save_vertices_indxs = False

        # Setting other parameters
        numHGScales = 4  # How many times to downsample inside each HourGlass

        # Network definition
        net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
        if gpu_id >= 0:
            torch.cuda.set_device(device=gpu_id)
            net.cuda()

    else:
        raise RuntimeError('undefined network')

    return net, p

def pred_valid_set(w, wo):
    # Setting other parameters
    gpu_id = 0  # Select which GPU, -1 if CPU
    # modelName = tb.construct_name(p, "HourGlass")
    # Define the Network and load the pre-trained weights as a CPU tensor
    [net, p] = BuildNet(model)

    net.load_state_dict(torch.load(modelName, map_location=lambda storage, loc: storage))
    net.eval()

    # No need to back-propagate
    for par in net.parameters():
        par.requires_grad = False

    # Transfer to GPU if needed
    if gpu_id >= 0:
        torch.cuda.set_device(device=gpu_id)
        net.cuda()

    num_patches_per_image = 50
    num_images = 14

    with_road_max = {}  # save the max prediction of road center in every validate patch with road inside
    without_road_max = {}  # save the max prediction of road center in every validate patch without road inside

    with_road_max_value = 0
    with_road_min_value = 10000

    without_road_max_value = 0
    without_road_min_value = 10000

    with_road_max_preds = []
    without_road_max_preds = []

    imagefiles = glob.glob(os.path.join(db_root_dir, '*_img.png'))
    count = 50
    indx = 0
    for imagefile in imagefiles:
        print(imagefile)

        img = Image.open(imagefile)
        img = np.array(img, dtype=np.float32)

        checkfile = imagefile[:-7] + 'chk.png'
        chk = Image.open(checkfile)
        chk = np.asarray(chk, np.float32)
        chk = chk.transpose((2, 0, 1))

        if len(img.shape) == 2:
            image_tmp = img
            h, w = image_tmp.shape
            img = np.zeros((h, w, 3))
            img[:, :, 0] = image_tmp
            img[:, :, 1] = image_tmp
            img[:, :, 2] = image_tmp
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        inputs = img / 255 - 0.5

        # Forward pass of the mini-batch
        inputs = Variable(inputs)
        if gpu_id >= 0:
            inputs = inputs.cuda()

        output = net.forward(inputs)
        if model == 'Hourglass':
            pred = np.squeeze(np.transpose(output[len(output) - 1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))
        elif model == 'UNet':
            pred = np.squeeze(np.transpose(output.cpu().data.numpy()[0, :, :, :], (1, 2, 0)))
        else:
            raise RuntimeError('model unimplemented')


        centers_idx = [np.argmax(pred)]

        pred_mask = tb.make_gt(None, centers_idx, p['inputRes'], 2)

        chk[1, :, :] = np.squeeze(pred_mask * 255, axis=2)  # set blue
        chk = chk.transpose((1, 2, 0))

        _, outputfile = os.path.split(imagefile)
        scipy.misc.imsave(output_dir + outputfile[:-7] + '_pred_chk.png', chk)

        # with open(output_dir + 'img_%02d_patch_%02d_pred_peaks.json' %(ii+1, jj+1), 'w') as outfile:
        #     json.dump(data, outfile)

        have_road = int(outputfile[outputfile.find('r') + 1])

        maxvalue = np.max(pred)
        maxvalue = float(maxvalue)
        if have_road == 1:
            with_road_max[outputfile] = maxvalue
            with_road_max_preds.append(maxvalue)
            if with_road_max_value < maxvalue:
                with_road_max_value = maxvalue
            if with_road_min_value > maxvalue:
                with_road_min_value = maxvalue

        else:
            without_road_max[outputfile] = maxvalue
            without_road_max_preds.append(maxvalue)
            if without_road_max_value < maxvalue:
                without_road_max_value = maxvalue
            if without_road_min_value > maxvalue:
                without_road_min_value = maxvalue

        # indx += 1
        # if indx>=count:
        #     break

    with_road_max['max'] = with_road_max_value
    with_road_max['min'] = with_road_min_value
    with_road_max['maxpreds'] = with_road_max_preds

    without_road_max['max'] = without_road_max_value
    without_road_max['min'] = without_road_min_value
    without_road_max['maxpreds'] = without_road_max_preds

    w = open(w, 'w')
    json.dump(with_road_max, w)

    wo = open(wo, 'w')
    json.dump(without_road_max, wo)


def road_center_pred_stat(w, wo):
    with_road_json_file = w
    without_road_json_file = wo

    with open(with_road_json_file) as inputfile:
        with_road_pred_obj = json.load(inputfile)

    with open(without_road_json_file) as inputfile:
        without_road_pred_obj = json.load(inputfile)

    print('with_road_pred_min: {}'.format(with_road_pred_obj['min']))
    print('without_road_pred_max: {}'.format(without_road_pred_obj['max']))

    with_road_pred_max_values = with_road_pred_obj['maxpreds']
    without_road_pred_max_values = without_road_pred_obj['maxpreds']

    with_road_mean = np.mean(with_road_pred_max_values)
    without_road_mean = np.mean(without_road_pred_max_values)

    with_road_std = np.std(with_road_pred_max_values)
    without_road_std = np.std(without_road_pred_max_values)

    print('width_road_mean={}, with_road_std={}'.format(with_road_mean, with_road_std))
    print('widthout_road_mean={}, without_road_std={}'.format(without_road_mean, without_road_std))

    with_road_pred_max_values = np.sort(with_road_pred_max_values)
    print(with_road_pred_max_values)
    without_road_pred_max_values = np.sort(without_road_pred_max_values)
    print(without_road_pred_max_values)

    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    num_bins = 30
 

    plt.figure()
    plt.subplot(121)
    n, bins, patches = plt.hist(with_road_pred_max_values, num_bins,  facecolor='blue', alpha=0.5)
    # y = mlab.normpdf(bins, with_road_mean, with_road_std)
    # plt.plot(bins, y, 'r--')
    plt.xlabel('The max confidence')
    plt.ylabel('Number of samples')
    plt.title(r'w roads: $\mu={}$, $\sigma={}$'.format(format(with_road_mean, '.3f'), format(with_road_std, '.3f')))
    plt.subplots_adjust(left=0.15)

    plt.subplot(122)
    n, bins, patches = plt.hist(without_road_pred_max_values, num_bins,  facecolor='red', alpha=0.5)
    # yout = mlab.normpdf(bins, without_road_mean, without_road_std)
    # plt.plot(bins, yout, 'r--')
    plt.xlabel('The max confidence')
    plt.ylabel('Number of samples')
    plt.title(
        r'w/o roads: $\mu={}$, $\sigma={}$'.format(format(without_road_mean, '.3f'), format(without_road_std, '.3f')))
    plt.subplots_adjust(left=0.15)

    plt.show()

if __name__ == '__main__':
    w = os.path.join(output_dir, 'prediction_with_road.json')
    wo = os.path.join(output_dir, 'prediction_without_road.json')
    # pred_valid_set(w, wo)
    road_center_pred_stat(w, wo)
