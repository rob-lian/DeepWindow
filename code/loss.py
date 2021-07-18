import torch
import numpy as np
import torch.nn.functional as F

def WCE_loss(pred, target):
    eps = 1e-10
    loss = target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps)
    loss = torch.pow(np.e, target) * loss

    return -torch.mean(loss)
