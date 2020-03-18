import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

def mixup_data(x, y, alpha):
    # https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/utils.py
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        if lam < 0.5:
            lam = 1 - lam
    else:
        lam = 1.

    if type(x) != tuple:
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index,:]
    else:
        batch_size = x[0].size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = []
        for _x in x:
            mixed_x.append(lam * _x + (1 - lam) * _x[index,:])
        mixed_x = tuple(mixed_x)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class CalibrationMixup(nn.Module):
    def __init__(self, layer_number=None):
        super(CalibrationMixup, self).__init__()
        self.p = Parameter(torch.zeros(1))
        self.layer_number = layer_number

    def forward(self, x, y, alpha):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
            if lam < 0.5:
                lam = 1 - lam
        else:
            lam = 1.

        # caliblation
        eps = 1e-6
        m1_lam = np.clip((1 - lam), eps, 1 - eps)
        #exponent = torch.exp(self.p)
        exponent = torch.log(1 + torch.exp(self.p) * (np.exp(1) - 1))
        lam_calib = torch.clamp(1 - torch.pow(2 * m1_lam, exponent) / 2, eps, 1 - eps)

        if type(x) != tuple:
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()

            #mixed_x = lam * x + (1 - lam) * x[index,:]
            mixed_x = lam_calib * x + (1 - lam_calib) * x[index,:]
        else:
            batch_size = x[0].size()[0]
            index = torch.randperm(batch_size).cuda()

            mixed_x = []
            for _x in x:
                #mixed_x.append(lam * _x + (1 - lam) * _x[index,:])
                mixed_x.append(lam_calib * _x + (1 - lam_calib) * _x[index,:])
            mixed_x = tuple(mixed_x)

        y_a, y_b = y, y[index]
        #return mixed_x, y_a, y_b, lam_calib
        return mixed_x, y_a, y_b, lam

def CrossEntropyLossForMixup(num_class=100, label_smooth=0.0):
    def loss_func(input, y_a, y_b, lam):
        soft_target = _get_mixed_soft_target(y_a, y_b, lam, num_class=num_class, label_smooth=0.0)
        loss = _CrossEntropyLossWithSoftTarget(input, soft_target)
        return loss
    return loss_func

def _get_mixed_soft_target(y_a, y_b, lam, num_class=100, label_smooth=0.0):
    onehot_a = (torch.eye(num_class)[y_a] * (1- label_smooth) + label_smooth / num_class).cuda()
    onehot_b = (torch.eye(num_class)[y_b] * (1- label_smooth) + label_smooth / num_class).cuda()
    return lam * onehot_a + (1 - lam) * onehot_b

def _CrossEntropyLossWithSoftTarget(input, soft_target):
    loss = - torch.mean(torch.sum(F.log_softmax(input, dim=1) * soft_target, dim=1))
    return loss

