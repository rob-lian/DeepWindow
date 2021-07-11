import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, target, output, global_step):

        grid_image = make_grid(image[:,:,:,:].clone().cpu().data, 12, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(torch.from_numpy(output.detach().cpu().numpy()), 12, normalize=True, range=(0, 255))

        writer.add_image('Predicted label', grid_image, global_step, dataformats='CHW')
        grid_image = make_grid(torch.from_numpy(target.detach().cpu().numpy()), 12, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
