# Includes
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import roads.HourGlass as nt
import roads.UNet as un
import timeit
import scipy.misc
from PIL import Image
import roads.patch.bifurcations_toolbox_roads as tbroads
from tqdm import tqdm
from roads.patch.summaries import TensorboardSummary
import glob
import cv2

# Setting of parameters
dsname = 'DRIVE'
db_root_dir = '/data/DRIVE/'
save_dir = '../results/' #Directory where the model will be stored during the training

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
        p['useAug'] = 0  # Random flip del:Use Random rotations in [-30, 30] and scaling in [.75, 1.25]
        p['useHist'] = 0 # rgb2gray combining Histogram equalization
        # mass uses 64*64, deepglobe uses 128*128
        p['inputRes'] = (64, 64)  # Input Resolution # default 64
        p['outputRes'] = (64, 64)  # Output Resolution (same as input)
        p['g_size'] = 32  # Higher means narrower Gaussian # default 64
        p['trainBatch'] = 4  # Number of Images in each mini-batch # default 64
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

if torch.cuda.is_available():
    gpu_id = 0
else:
    gpu_id = -1

snapshot = 10  # Store a model every snapshot epochs
useVal = 1  # See evolution of the test set when training?
nTestInterval = 1  # Run on test set every nTestInterval iterations

model = "Hourglass"
[net, p] = BuildNet(model)

# Loss function definition
criterion = nn.MSELoss(size_average=True, reduce=True)

# Use the following optimizer
optimizer = optim.RMSprop(net.parameters(), lr=1e-5, alpha=0.99, momentum=0.0)


# Preparation of the data loaders
# Define augmentation transformations as a composition
trans = []

if p['useHist'] == 1:
    trans.append(tbroads.GrayHistEqual())
if p['useAug'] == 1:
   # trans.append(tbroads.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)))
    trans.append(tbroads.RandomFlip())

trans.append(tbroads.ToTensor())

composed_transforms = transforms.Compose(trans)

db_train = tbroads.ToolDataset(train=True, online=True, have_road=True, inputRes=p['inputRes'], outputRes=p['outputRes'],
                          sigma=float(p['outputRes'][0]) / p['g_size'],
                          db_root_dir=db_root_dir, transform=composed_transforms,
                               save_vertices_indxs=False, dsname=dsname)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True)


val_dir = '\data\Mass_Point_Supervision\val_gt\'
val_file_list = glob.glob(os.path.join(val_dir,'*_img.png'))

num_img_tr = len(trainloader)

running_loss_tr = 0
running_loss_val = 0
loss_tr = []
loss_val = []


modelName = tbroads.construct_name(p, model)


print("Training Network")

run_dir = os.path.join(save_dir, 'run/'+dsname+'/'+model+'/')
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

runs = sorted(glob.glob(os.path.join(run_dir, 'experiment_*')))
run_id = 0
for folder in runs:
    id = int(folder.split('_')[-1])+1
    run_id = id if id > run_id else run_id

experiment_dir = os.path.join(run_dir, 'experiment_{}'.format(str(run_id)))
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

summary = TensorboardSummary(experiment_dir)
writer = summary.create_summary()
# Main Training and Testing Loop

start_epoch = 0
nEpochs = 600
minValidLoss = 1e10

for epoch in range(start_epoch+1, nEpochs):
    start_time = timeit.default_timer()

    net.train()
    # One training epoch
    train_loss = 0.0
    val_loss = 0.0
    tbar = tqdm(trainloader)
    for ii, sample_batched in enumerate(tbar):
        img, gt, mask_gt = sample_batched['image'], sample_batched['gt'], sample_batched['mask_gt']

        inputs = img / 255 - 0.5
        if model == 'Hourglass':
            gts = 255 * gt # gt : 0 ~ 1
        elif model == 'UNet':
            gts = gt
        else:
            raise RuntimeError('net unimplemented')

        inputs, gts = Variable(inputs), Variable(gts)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        optimizer.zero_grad()
        outputs = net.forward(inputs)


        if model == 'Hourglass':
            losses = [None] * p['numHG']
            for i in range(0, len(outputs)):
                losses[i] = criterion(outputs[i], gts)
            loss = sum(losses)

        else:
            raise RuntimeError('undefined network')

        train_loss += loss.item()

        tbar.set_description('epoch: %d, Train MSEloss: %.3f' % (epoch, train_loss / (ii + 1)))
        writer.add_scalar('train/total_loss_iter', loss.item(), ii + num_img_tr * epoch)

        if num_img_tr>10:
            image_interval = num_img_tr // 10
        else:
            image_interval = 1

        if (ii+1) % image_interval == 0:
            global_step = ii + num_img_tr * epoch
            if model=='Hourglass':
                summary.visualize_image(writer, img, gts, outputs[1], global_step)
            else:
                summary.visualize_image(writer, img, gts, outputs, global_step)
            # print('summary visualize, global_step={}'.format(global_step))

        loss.backward()
        optimizer.step()

    writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

    # Save the model
    if ((epoch+1) % snapshot) == 0 and epoch != 0:
        checkpoint = os.path.join(experiment_dir, modelName+'_epoch-'+str(epoch)+'.pth')
        torch.save(net.state_dict(), checkpoint)
        print('model saved:', checkpoint)


    # One testing epoch
    if useVal:
        if epoch % nTestInterval == (nTestInterval-1):

            if dsname == "mass":
                num_patches_per_image = 50
                num_images = 14
            elif dsname == 'deepglobe':
                num_patches_per_image = 10
                num_images = 1000
            elif dsname == 'DRIVE':
                num_patches_per_image = 50
                num_images = 5

            num_img_val = num_patches_per_image*num_images
            net.eval()
            for val_file in val_file_list:

                img = Image.open(val_file)
                img = np.asarray(img)

                if p['useHist'] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.equalizeHist(img)

                img = img.astype(np.float32)

                if len(img.shape) == 2:
                    image_tmp = img
                    h, w = image_tmp.shape
                    img = np.zeros((h, w, 3), dtype=np.float32)
                    img[:,:,0] = image_tmp
                    img[:,:,1] = image_tmp
                    img[:,:,2] = image_tmp

                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img)
                img = img.unsqueeze(0)

                inputs = img / 255 - 0.5

                gt_file = val_file[:-7] + 'gt.png'
                gt = Image.open(gt_file)
                gt = np.array(gt)
                if len(gt.shape) == 2:
                    gt_tmp = gt
                    h, w = gt_tmp.shape
                    gt = np.zeros((h, w, 1), np.float32)
                    gt[:,:,0] = gt_tmp

                gt = gt.transpose((2, 0, 1))
                gt = torch.from_numpy(gt)

                gt = Variable(gt)
                if gpu_id >= 0:
                    gt = gt.cuda()

                # Forward pass of the mini-batch
                inputs = Variable(inputs)
                if gpu_id >= 0:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    outputs = net.forward(inputs)

                if model == 'Hourglass':
                    losses = [None] * p['numHG']
                    for i in range(0, len(outputs)):
                        losses[i] = criterion(outputs[i], gts)
                    loss = sum(losses)
                else:
                    raise RuntimeError('undefined network')


                val_loss += loss.item()

            if minValidLoss>val_loss:
                minValidLoss = val_loss
                checkpoint = os.path.join(experiment_dir, modelName+'_epoch-'+str(epoch)+'_best.pth')
                torch.save(net.state_dict(), checkpoint)
                print('best model saved:', checkpoint)

            writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
            print('val_loss = {}'.format(val_loss))
