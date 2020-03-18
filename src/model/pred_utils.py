import torch
import torch.nn as nn
from albumentations.augmentations import functional as albF

from tqdm import tqdm
import numpy as np
import cv2

def predict_logit(net, loader):
    net = net.cuda()
    net.eval()
    
    gra = []
    vow = []
    con = []
    
    print('predict logit')
    with torch.no_grad():
        for batch_idx, imgs in enumerate(tqdm(loader)):
            imgs = imgs.cuda()

            outputs = net(imgs)
 
            gra.append(outputs[0].cpu().numpy())
            vow.append(outputs[1].cpu().numpy())
            con.append(outputs[2].cpu().numpy())
            
    gra = np.concatenate(gra)
    vow = np.concatenate(vow)
    con = np.concatenate(con)
    
    print('gra', gra.shape)
    print('vow', vow.shape)
    print('con', con.shape)
    
    return gra, vow, con

def logit3_to_label(logit3):
    """
    Args: tuple (logit0, logit1, logit2)
    """
    label0 = np.argmax(logit3[0], axis=1)
    label1 = np.argmax(logit3[1], axis=1)
    label2 = np.argmax(logit3[2], axis=1)
    return label0, label1, label2

class TTA:
    def __init__(self, net, transform, degrees=[0], scales=[0], shifts=[[0,0]], low_thresh=None):
        self.net = net
        self.transform = transform
        self.low_thresh = low_thresh

        self.tta_args = self.get_tta_args(degrees, scales, shifts)

    def get_tta_args(self, degrees, scales, shifts):
        tta_args = []
        tta_args.append({'degree':0.0, 'scale':0.0, 'shift':[0.0,0.0]})

        for deg in degrees:
            if deg != 0:
                tta_args.append({'degree':deg, 'scale':0.0, 'shift':[0.0,0.0]})

        for sca in scales:
            if sca != 0:
                tta_args.append({'degree':0.0, 'scale':sca, 'shift':[0.0,0.0]})

        for shi in shifts:
            if shi[0] != 0 and shi[1] != 0:
                tta_args.append({'degree':0.0, 'scale':0.0, 'shift':shi})

        return tta_args

    @staticmethod
    def shift_scale_rotate(degree, scale, shift):
        def func(img):
            return albF.shift_scale_rotate(img, degree, 1.0 + scale, shift[0], shift[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return func

    def predict(self, images, batch_size):
        preds = None

        for tta_arg in self.tta_args:
            ds = ImgDataset(images, self.transform, TTA.shift_scale_rotate(**tta_arg))
            loader = get_dataloader(ds, batch_size)

            output = predict_logit(self.net, loader)
            if preds is None:
                preds = list(output)
                if self.low_thresh is not None:
                    for i in range(len(preds)):
                        preds[i] = np.maximum(preds[i], self.low_thresh)

            else:
                for i, oup in enumerate(output):
                    if self.low_thresh is None:
                        preds[i] += oup
                    else:
                        preds[i] += np.maximum(oup, self.low_thresh)


        for i in range(len(preds)):
            preds[i] = preds[i] / len(self.tta_args)

        return preds

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform=None, add_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.add_transform = add_transform

    def __getitem__(self, idx):
        img = self.imgs[idx]

        if self.add_transform is not None:
            img = self.add_transform(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img

    def __len__(self):
        return len(self.imgs)

def get_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

