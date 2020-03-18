from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
import torch
import math

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_min=1, p_max=None):
        super(GeM,self).__init__()

        self.p = Parameter(torch.ones(1)*p) if p is not None else None
        if p is None:
            self.pooling = GlobalAvePooling()

        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max

    def forward(self, x):
        if self.p is not None:
            p = torch.clamp(self.p, self.p_min) if self.p_max is None else torch.clamp(self.p, self.p_min, self.p_max)
            return GeM.gem(x, p=p, eps=self.eps)
        else:
            return self.pooling(x)

    @staticmethod
    def gem(x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class GlobalAvePooling(nn.Module):
    def __init__(self):
        super(GlobalAvePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return x

class MaskingPooling(nn.Module):
    def __init__(self, h, w, in_channels, num_masks, reduction, use_depthwise=True):
        super(MaskingPooling, self).__init__()

        n_hidden = in_channels // reduction

        if use_depthwise:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=(h, w), stride=1, groups=in_channels), 
                                       nn.Conv2d(in_channels, n_hidden, kernel_size=1, stride=1))
        else:
            self.conv1 = nn.Conv2d(in_channels, n_hidden, kernel_size=(h, w), stride=1)
        
        
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(n_hidden, num_masks)
        self.softmax = nn.Softmax()

        self.masks = Parameter(torch.Tensor(1, num_masks, h, w))
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.masks, a=math.sqrt(5))

    def forward(self, x):
        y = self.conv1(x)
        y = y.view(y.size(0), y.size(1))
        y = self.bn1(y)
        y = self.relu(y)
        y = self.fc1(y)
        y = self.softmax(y)

        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * self.masks
        y = torch.sum(y, dim=1, keepdim=True)

        y = x * y
        y = torch.sum(y, dim=(2, 3))

        return y