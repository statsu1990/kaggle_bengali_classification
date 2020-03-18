from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch

import math

class GlobalAvePooling(nn.Module):
    def __init__(self):
        super(GlobalAvePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return x

class Mix_v1(nn.Module):
    def __init__(self, x_lengths, num_output, dropout_p=0.0):
        super(Mix_v1, self).__init__()
        
        num_x = len(x_lengths)

        self.converts = nn.ModuleList()
        for i in range(num_x):
            self.converts.append(nn.Sequential(GlobalAvePooling(), 
                                               nn.Dropout(dropout_p),
                                               nn.Linear(x_lengths[i], num_output),
                                               ))

        self.weight = Parameter(torch.Tensor(1, num_output, num_x))
        self.softmax = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, xs):

        ys = []
        for i in range(len(xs)):
            ys.append(torch.unsqueeze(self.converts[i](xs[i]), 2))
        ys = torch.cat(ys, dim=2)

        w = self.softmax(self.weight)
        y = torch.sum(w * ys, dim=2)

        return y


