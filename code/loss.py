import torch
import numpy as np
import torch.nn.functional as F

def WCE_loss(pred, target):
    eps = 1e-10
    # return torch.mean(torch.pow((pred - target), 2) * torch.pow(4096, target))
    # return -torch.mean(target * torch.log(pred+eps) + (1-target) * torch.log(1-pred+eps))
    # print('pred:', np.any(np.isnan(pred.detach().cpu().numpy())))
    # print('target:', np.any(np.isnan(target.detach().cpu().numpy())))
    # exit()

    loss = target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps)
    loss = torch.pow(np.e, target) * loss

    # print('loss:', np.any(np.isnan(loss.detach().cpu().numpy())))


    return -torch.mean(loss)
