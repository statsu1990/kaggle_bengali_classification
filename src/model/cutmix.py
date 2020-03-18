"""
https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py

MIT License

Copyright (c) 2019 hysts

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cutmix_data(x, y, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        if lam < 0.5:
            lam = 1 - lam
    else:
        lam = 1.

    if type(x) != tuple:
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        image_h, image_w = x.size()[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        mixed_x = x
        mixed_x[:, :, y0:y1, x0:x1] = x[index, :, y0:y1, x0:x1]

    else:
        # x[0]: (B, C, W, H)
        # x[1] ~ x[N]: (B, ?)
        batch_size = x[0].size()[0]
        index = torch.randperm(batch_size).cuda()

        image_h, image_w = x[0].size()[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        mix_x = x[0]
        mix_x[:, :, y0:y1, x0:x1] = x[0][index, :, y0:y1, x0:x1]

        mixed_x = [mix_x]
        for i in range(len(x)):
            if i > 0:
                mixed_x.append(lam * x[i] + (1 - lam) * x[i][index,:])
        mixed_x = tuple(mixed_x)

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam