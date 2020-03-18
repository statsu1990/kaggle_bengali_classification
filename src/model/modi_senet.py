import torch
import torch.nn as nn
import copy
import random

from model import mixup
from model import cutmix
from model import mish

from dropblock import DropBlock2D

def copy_model(mdl):
    #mdl_copy = type(mdl)() # get a new instance
    #mdl_copy.load_state_dict(mdl.state_dict()) # copy weights and stuff

    mdl_copy = copy.deepcopy(mdl)
    mdl_copy.load_state_dict(mdl.state_dict()) # copy weights and stuff

    return mdl_copy

def first_conv2d_3ch_to_1ch(layer0, input_3x3=False):
    # https://www.kaggle.com/c/bengaliai-cv19/discussion/130311
    if input_3x3:
        out_channel = 64
        kernel_size = 3
        stride = 2
        padding = 1
        bias = False
    else:
        out_channel = 64
        kernel_size = 7
        stride = 2
        padding = 3
        bias = False

    #converting to list
    arch = list(layer0.children())
    #saving the weights of the forst conv in w
    w = arch[0].weight
    #creating new Conv2d to accept 1 channel 
    arch[0] = nn.Conv2d(1, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    #substituting weights of newly created Conv2d with w from but we have to take mean
    #to go from  3 channel to 1
    arch[0].weight = nn.Parameter(torch.sum(w, dim=1, keepdim=True))
    arch = nn.Sequential(*arch)

    return arch

def state_mixup_cutmix(training, target, 
                           mixup_alpha, mix_cand_layers, 
                           cutmix_alpha, cutmix_cand_layers, 
                           mixup_p=0.5):
        """
        Returns:
            do_mix, mixfunc, alpha, mix_layer
        """
        can_mixup = training and (target is not None) and (mixup_alpha is not None) and (mix_cand_layers is not None)
        can_cutmix = training and (target is not None) and (cutmix_alpha is not None) and (cutmix_cand_layers is not None)

        do_mix = can_mixup or can_cutmix

        if can_mixup and can_cutmix:
            do_mixup = (random.random() < mixup_p)
        elif can_mixup:
            do_mixup = True
        elif can_cutmix:
            do_mixup = False
        else:
            do_mixup = None

        if do_mix and do_mixup:
            alpha = mixup_alpha
            mix_layer = random.choice(mix_cand_layers)
            mixfunc = mixup.mixup_data
        elif do_mix and not do_mixup:
            alpha = cutmix_alpha
            mix_layer = random.choice(cutmix_cand_layers)
            mixfunc = cutmix.cutmix_data
        else:
            alpha = None
            mix_layer = None
            mixfunc = None

        return do_mix, mixfunc, alpha, mix_layer

def calc_layer(xs, layer_module, gra_attnt, vow_attnt, con_attnt, num_x_is_3):
    use_attention = gra_attnt is not None

    if num_x_is_3:
        next_num_x_is_3 = True

        gra_x = layer_module(xs[0])
        vow_x = layer_module(xs[1])
        con_x = layer_module(xs[2])

        if use_attention:
            gra_x = gra_attnt(gra_x)
            vow_x = vow_attnt(vow_x)
            con_x = con_attnt(con_x)

        return (gra_x, vow_x, con_x), next_num_x_is_3

    else:
        x = layer_module(xs)

        if use_attention:
            next_num_x_is_3 = True

            gra_x = gra_attnt(x)
            vow_x = vow_attnt(x)
            con_x = con_attnt(x)
            return (gra_x, vow_x, con_x), next_num_x_is_3
        else:
            next_num_x_is_3 = False
            return x, next_num_x_is_3

class SENetEncoder(nn.Module):
    def __init__(self, senet, input3ch=True):
        super(SENetEncoder, self).__init__()
        self.layer0 = senet.layer0
        if not input3ch:
            self.layer0 = first_conv2d_3ch_to_1ch(self.layer0)
        self.layer1 = senet.layer1
        self.layer2 = senet.layer2
        self.layer3 = senet.layer3
        self.layer4 = senet.layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze_first_layer(self, freeze=True):
        for param in self.layer0.parameters():
            param.requires_grad = not freeze
        return

class SENetEncoder_Mixup(nn.Module):
    def __init__(self, senet, input3ch=True, three_neck=False, 
                 mixup_alpha=None, mix_cand_layers=None, use_mish=False, cutmix_alpha=None, cutmix_cand_layers=None, 
                 output_layer3=False, output_layer2=False, dropblock_p=None, upsample_size=None, calib_mixup=False):
        super(SENetEncoder_Mixup, self).__init__()
        self.three_neck = three_neck
        self.mixup_alpha = mixup_alpha
        self.mix_cand_layers = mix_cand_layers
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_cand_layers = cutmix_cand_layers
        self.output_layer3 = output_layer3
        self.output_layer2 = output_layer2

        self.layer0 = senet.layer0
        if not input3ch:
            self.layer0 = first_conv2d_3ch_to_1ch(self.layer0)

        self.layer1 = senet.layer1
        self.layer2 = senet.layer2
        self.layer3 = senet.layer3
        if three_neck:
            self.layer4_1 = copy_model(senet.layer4)
            self.layer4_2 = copy_model(senet.layer4)
            self.layer4_3 = copy_model(senet.layer4)
        else:
            self.layer4 = copy_model(senet.layer4)

        self.calb_mixup0 = mixup.CalibrationMixup(layer_number=0) if calib_mixup else None
        self.calb_mixup1 = mixup.CalibrationMixup(layer_number=1) if calib_mixup else None
        self.calb_mixup2 = mixup.CalibrationMixup(layer_number=2) if calib_mixup else None
        self.calb_mixup3 = mixup.CalibrationMixup(layer_number=3) if calib_mixup else None

        self.dropblock0 = DropBlock2D(drop_prob=dropblock_p, block_size=10) if dropblock_p is not None else None
        self.dropblock1 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None
        self.dropblock2 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None

        self.upsample = nn.Upsample(size=upsample_size, mode='bilinear', align_corners=True) if upsample_size is not None else None

        if use_mish:
            mish.relu_to_mish(self.layer0)
            mish.relu_to_mish(self.layer1)
            mish.relu_to_mish(self.layer2)
            mish.relu_to_mish(self.layer3)

            if three_neck:
                mish.relu_to_mish(self.layer4_1)
                mish.relu_to_mish(self.layer4_2)
                mish.relu_to_mish(self.layer4_3)
            else:
                mish.relu_to_mish(self.layer4)

    def forward(self, x, target=None):

        do_mix, mixfunc, alpha, mix_layer = state_mixup_cutmix(self.training, target, self.mixup_alpha, self.mix_cand_layers, self.cutmix_alpha, self.cutmix_cand_layers)

        if self.upsample is not None:
            x = self.upsample(x)

        if do_mix:

            if mix_layer == 0:
                if mixfunc == mixup.mixup_data:
                    if self.calb_mixup0 is not None:
                        mixfunc = self.calb_mixup0
                x, y_a, y_b, lam = mixfunc(x, target, alpha)

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if mix_layer == 1:
                if mixfunc == mixup.mixup_data:
                    if self.calb_mixup1 is not None:
                        mixfunc = self.calb_mixup1
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock1 is not None:
                x = self.dropblock1(x)

            x = self.layer2(x)
            if mix_layer == 2:
                if mixfunc == mixup.mixup_data:
                    if self.calb_mixup2 is not None:
                        mixfunc = self.calb_mixup2
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock2 is not None:
                x = self.dropblock2(x)
            if self.output_layer2:
                return x, y_a, y_b, lam

            x = self.layer3(x)
            if mix_layer == 3:
                if mixfunc == mixup.mixup_data:
                    if self.calb_mixup3 is not None:
                        mixfunc = self.calb_mixup3
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.output_layer3:
                return x, y_a, y_b, lam

            if self.three_neck:
                x1 = self.layer4_1(x)
                x2 = self.layer4_2(x)
                x3 = self.layer4_3(x)
                return (x1, x2, x3), y_a, y_b, lam
            else:
                x = self.layer4(x)
                return x, y_a, y_b, lam

        else:
            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if self.dropblock1 is not None:
                x = self.dropblock1(x)

            x = self.layer2(x)
            if self.dropblock2 is not None:
                x = self.dropblock2(x)
            if self.output_layer2:
                return x

            x = self.layer3(x)
            if self.output_layer3:
                return x

            if self.three_neck:
                x1 = self.layer4_1(x)
                x2 = self.layer4_2(x)
                x3 = self.layer4_3(x)
                return x1, x2, x3
            else:
                x = self.layer4(x)
                return x

    def freeze_layer(self, freeze=True, target_layers=[0, 1, 2]):
        if 0 in target_layers:
            for param in self.layer0.parameters():
                param.requires_grad = not freeze

        if 1 in target_layers:
            for param in self.layer1.parameters():
                param.requires_grad = not freeze

        if 2 in target_layers:
            for param in self.layer2.parameters():
                param.requires_grad = not freeze

        if 3 in target_layers:
            for param in self.layer3.parameters():
                param.requires_grad = not freeze

        if 4 in target_layers:
            for param in self.layer4.parameters():
                param.requires_grad = not freeze

        return

class SENetEncoder_CalibMixup_Multiscale_v2(nn.Module):
    def __init__(self, senet, input3ch=True, three_neck=False, 
                 mixup_alpha=None, mix_cand_layers=None, use_mish=False, cutmix_alpha=None, cutmix_cand_layers=None, 
                 output_layers=[2,3,4], dropblock_p=None):
        super(SENetEncoder_CalibMixup_Multiscale_v2, self).__init__()
        self.three_neck = False
        self.mixup_alpha = mixup_alpha
        self.mix_cand_layers = mix_cand_layers
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_cand_layers = cutmix_cand_layers
        self.output_layers = output_layers

        self.layer0 = senet.layer0
        if not input3ch:
            self.layer0 = first_conv2d_3ch_to_1ch(self.layer0)

        self.layer1 = senet.layer1 # (256, 32, 32)
        self.layer2 = senet.layer2 # (512, 16, 16)
        self.layer3 = senet.layer3 # (1024, 8, 8)
        self.layer4 = senet.layer4 # (2048, 4, 4)

        self.calb_mixup0 = mixup.CalibrationMixup(layer_number=0)
        self.calb_mixup1 = mixup.CalibrationMixup(layer_number=1)
        self.calb_mixup2 = mixup.CalibrationMixup(layer_number=2)
        self.calb_mixup3 = mixup.CalibrationMixup(layer_number=3)

        self.dropblock0 = DropBlock2D(drop_prob=dropblock_p, block_size=10) if dropblock_p is not None else None
        self.dropblock1 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None
        self.dropblock2 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None

        if use_mish:
            mish.relu_to_mish(self.layer0)
            mish.relu_to_mish(self.layer1)
            mish.relu_to_mish(self.layer2)
            mish.relu_to_mish(self.layer3)
            mish.relu_to_mish(self.layer4)

        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, target=None):

        do_mix, mixfunc, alpha, mix_layer = state_mixup_cutmix(self.training, target, self.mixup_alpha, self.mix_cand_layers, self.cutmix_alpha, self.cutmix_cand_layers)

        if do_mix:
            output = []

            if mix_layer == 0:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup0
                x, y_a, y_b, lam = mixfunc(x, target, alpha)

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if mix_layer == 1:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup1
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock1 is not None:
                x = self.dropblock1(x)
            if 1 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer2(x)
            if mix_layer == 2:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup2
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock2 is not None:
                x = self.dropblock2(x)
            if 2 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer3(x)
            if mix_layer == 3:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup3
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if 3 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer4(x)
            if 4 in self.output_layers:
                output.append(self.pooling(x))

            return output, y_a, y_b, lam

        else:
            output = []

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if 1 in self.output_layers:
                output.append(self.pooling(x))
            if self.dropblock1 is not None:
                x = self.dropblock1(x)

            x = self.layer2(x)
            if 2 in self.output_layers:
                output.append(self.pooling(x))
            if self.dropblock2 is not None:
                x = self.dropblock2(x)

            x = self.layer3(x)
            if 3 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer4(x)
            if 4 in self.output_layers:
                output.append(self.pooling(x))

            return output

    def freeze_layer(self, freeze=True, target_layers=[0, 1, 2]):
        if 0 in target_layers:
            for param in self.layer0.parameters():
                param.requires_grad = not freeze

        if 1 in target_layers:
            for param in self.layer1.parameters():
                param.requires_grad = not freeze

        if 2 in target_layers:
            for param in self.layer2.parameters():
                param.requires_grad = not freeze

        if 3 in target_layers:
            for param in self.layer3.parameters():
                param.requires_grad = not freeze

        if 4 in target_layers:
            for param in self.layer4.parameters():
                param.requires_grad = not freeze

        return

class SENetEncoder_Multiscale_v2(nn.Module):
    def __init__(self, senet, input3ch=True, three_neck=False, 
                 mixup_alpha=None, mix_cand_layers=None, use_mish=False, cutmix_alpha=None, cutmix_cand_layers=None, 
                 output_layers=[2,3,4], dropblock_p=None):
        super(SENetEncoder_Multiscale_v2, self).__init__()
        self.three_neck = False
        self.mixup_alpha = mixup_alpha
        self.mix_cand_layers = mix_cand_layers
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_cand_layers = cutmix_cand_layers
        self.output_layers = output_layers

        self.layer0 = senet.layer0
        if not input3ch:
            self.layer0 = first_conv2d_3ch_to_1ch(self.layer0)

        self.layer1 = senet.layer1 # (256, 32, 32)
        self.layer2 = senet.layer2 # (512, 16, 16)
        self.layer3 = senet.layer3 # (1024, 8, 8)
        self.layer4 = senet.layer4 # (2048, 4, 4)

        self.dropblock0 = DropBlock2D(drop_prob=dropblock_p, block_size=10) if dropblock_p is not None else None
        self.dropblock1 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None
        self.dropblock2 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None

        if use_mish:
            mish.relu_to_mish(self.layer0)
            mish.relu_to_mish(self.layer1)
            mish.relu_to_mish(self.layer2)
            mish.relu_to_mish(self.layer3)
            mish.relu_to_mish(self.layer4)

        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, target=None):

        do_mix, mixfunc, alpha, mix_layer = state_mixup_cutmix(self.training, target, self.mixup_alpha, self.mix_cand_layers, self.cutmix_alpha, self.cutmix_cand_layers)

        if do_mix:
            output = []

            if mix_layer == 0:
                x, y_a, y_b, lam = mixfunc(x, target, alpha)

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if mix_layer == 1:
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock1 is not None:
                x = self.dropblock1(x)
            if 1 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer2(x)
            if mix_layer == 2:
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock2 is not None:
                x = self.dropblock2(x)
            if 2 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer3(x)
            if mix_layer == 3:
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if 3 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer4(x)
            if 4 in self.output_layers:
                output.append(self.pooling(x))

            return output, y_a, y_b, lam

        else:
            output = []

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if self.dropblock1 is not None:
                x = self.dropblock1(x)
            if 1 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer2(x)
            if self.dropblock2 is not None:
                x = self.dropblock2(x)
            if 2 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer3(x)
            if 3 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer4(x)
            if 4 in self.output_layers:
                output.append(self.pooling(x))

            return output

    def freeze_layer(self, freeze=True, target_layers=[0, 1, 2]):
        if 0 in target_layers:
            for param in self.layer0.parameters():
                param.requires_grad = not freeze

        if 1 in target_layers:
            for param in self.layer1.parameters():
                param.requires_grad = not freeze

        if 2 in target_layers:
            for param in self.layer2.parameters():
                param.requires_grad = not freeze

        if 3 in target_layers:
            for param in self.layer3.parameters():
                param.requires_grad = not freeze

        if 4 in target_layers:
            for param in self.layer4.parameters():
                param.requires_grad = not freeze

        return

class SENetEncoder_CalibMixup_Multiscale_v2(nn.Module):
    def __init__(self, senet, input3ch=True, three_neck=False, 
                 mixup_alpha=None, mix_cand_layers=None, use_mish=False, cutmix_alpha=None, cutmix_cand_layers=None, 
                 output_layers=[2,3,4], dropblock_p=None):
        super(SENetEncoder_CalibMixup_Multiscale_v2, self).__init__()
        self.three_neck = False
        self.mixup_alpha = mixup_alpha
        self.mix_cand_layers = mix_cand_layers
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_cand_layers = cutmix_cand_layers
        self.output_layers = output_layers

        self.layer0 = senet.layer0
        if not input3ch:
            self.layer0 = first_conv2d_3ch_to_1ch(self.layer0)

        self.layer1 = senet.layer1 # (256, 32, 32)
        self.layer2 = senet.layer2 # (512, 16, 16)
        self.layer3 = senet.layer3 # (1024, 8, 8)
        self.layer4 = senet.layer4 # (2048, 4, 4)

        self.calb_mixup0 = mixup.CalibrationMixup(layer_number=0)
        self.calb_mixup1 = mixup.CalibrationMixup(layer_number=1)
        self.calb_mixup2 = mixup.CalibrationMixup(layer_number=2)
        self.calb_mixup3 = mixup.CalibrationMixup(layer_number=3)

        self.dropblock0 = DropBlock2D(drop_prob=dropblock_p, block_size=10) if dropblock_p is not None else None
        self.dropblock1 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None
        self.dropblock2 = DropBlock2D(drop_prob=dropblock_p, block_size=5) if dropblock_p is not None else None

        if use_mish:
            mish.relu_to_mish(self.layer0)
            mish.relu_to_mish(self.layer1)
            mish.relu_to_mish(self.layer2)
            mish.relu_to_mish(self.layer3)
            mish.relu_to_mish(self.layer4)

        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, target=None):

        do_mix, mixfunc, alpha, mix_layer = state_mixup_cutmix(self.training, target, self.mixup_alpha, self.mix_cand_layers, self.cutmix_alpha, self.cutmix_cand_layers)

        if do_mix:
            output = []

            if mix_layer == 0:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup0
                x, y_a, y_b, lam = mixfunc(x, target, alpha)

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if mix_layer == 1:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup1
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock1 is not None:
                x = self.dropblock1(x)
            if 1 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer2(x)
            if mix_layer == 2:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup2
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if self.dropblock2 is not None:
                x = self.dropblock2(x)
            if 2 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer3(x)
            if mix_layer == 3:
                if mixfunc == mixup.mixup_data:
                    mixfunc = self.calb_mixup3
                x, y_a, y_b, lam = mixfunc(x, target, alpha)
            if 3 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer4(x)
            if 4 in self.output_layers:
                output.append(self.pooling(x))

            return output, y_a, y_b, lam

        else:
            output = []

            x = self.layer0(x)
            if self.dropblock0 is not None:
                x = self.dropblock0(x)

            x = self.layer1(x)
            if 1 in self.output_layers:
                output.append(self.pooling(x))
            if self.dropblock1 is not None:
                x = self.dropblock1(x)

            x = self.layer2(x)
            if 2 in self.output_layers:
                output.append(self.pooling(x))
            if self.dropblock2 is not None:
                x = self.dropblock2(x)

            x = self.layer3(x)
            if 3 in self.output_layers:
                output.append(self.pooling(x))

            x = self.layer4(x)
            if 4 in self.output_layers:
                output.append(self.pooling(x))

            return output

    def freeze_layer(self, freeze=True, target_layers=[0, 1, 2]):
        if 0 in target_layers:
            for param in self.layer0.parameters():
                param.requires_grad = not freeze

        if 1 in target_layers:
            for param in self.layer1.parameters():
                param.requires_grad = not freeze

        if 2 in target_layers:
            for param in self.layer2.parameters():
                param.requires_grad = not freeze

        if 3 in target_layers:
            for param in self.layer3.parameters():
                param.requires_grad = not freeze

        if 4 in target_layers:
            for param in self.layer4.parameters():
                param.requires_grad = not freeze

        return