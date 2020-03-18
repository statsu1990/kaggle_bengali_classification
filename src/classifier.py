import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import cv2
import matplotlib.pyplot as plt

import torch
import albumentations as alb
from albumentations.augmentations import transforms as albtr
from albumentations.pytorch import ToTensor as albToTensor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from data import data_utils as dt_utl
from image import image_proc as img_prc

from model import bengali_classifier
from model import senet
from model import modi_senet
from model import torch_data_utils as tdu
from model import train_utils as tru
from model import pred_utils as pru

pretrain_path_root = '../consideration/'

def get_checkpoint(path):
    cp = torch.load(path, map_location=lambda storage, loc: storage)
    return cp

def make_model_v0_4_1():
    """
    mixup, mish, cutmix, MultilabelStratifiedKFold, train_model_v2_1, class balance, SENetEncoder_Multiscale_v2+ClassifierModule_v9
    calib mixup, 
    """
    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = tdu.transform_wrapper(
                            [
                            ],
                        alb.Compose([
                            #albtr.RandomGamma(p=0.8),
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                            #        albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                            #        albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                            #        ], p=0.2),
                            #alb.OneOf([
                            #        #albtr.Blur(blur_limit=4, p=0.2),
                            #        albtr.MedianBlur(blur_limit=3, p=0.2)
                            #        ], p=0.1), 
                            #alb.OneOf([
                            #        GridMask(num_grid=(3,7), mode=0, rotate=15, p=1),
                            #        GridMask(num_grid=(3,7), mode=2, rotate=15, p=1),
                            #        ], p=0.7),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )
    ts_transformer = tdu.transform_wrapper(
                        [
                        ],
                        alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )

    ## model
    #encoder = modi_senet.SENetEncoder(senet.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder_ThreeNeck(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.2
    mix_cand_layers= [0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = 1.0
    cutmix_cand_layers = [0]
    output_layers = [2, 3, 4]
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    encoder_is_separately = three_neck
    encoder_use_mixup = True
    dropout_p = 0.0
    gem_p = None
    classifier_module = bengali_classifier.ClassifierModule_v9
    model = bengali_classifier.BengaliClassifier_v1(encoder, encoded_planes, 
                                                    encoder_is_separately, encoder_use_mixup, 
                                                    dropout_p, gem_p, 
                                                    classifier_module=classifier_module,
                                                    )

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.99, random_state=2020)
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    id = np.arange(len(labels))[:,None]

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
        if ifld == 0:
            print('training fold ', ifld)
            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            ## train with encoder freeze
            #tr_batch_size = 512
            #ts_batch_size = 512
            #tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            #vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            #
            #epochs = 2
            #lr = 1e-3 * tr_batch_size / 64
            #grad_accum_steps = 1
            #warmup_epoch=1
            #patience=5
            #factor=0.5
            #opt = 'AdaBound'
            #weight_decay=1e-4
            #loss_w = [0.5, 0.25, 0.25]
            #reference_label = labels[tr_idxs]
            #
            #model.freeze_encoder(freeze=True, target_layers=[0, 1, 2, 3, 4])
            #model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
            #                           epochs, lr, grad_accum_steps, 
            #                           warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label)
            #model.freeze_encoder(freeze=False)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 110
            lr = 1e-3
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            
            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v0_4_1_2():
    """
    mixup, mish, cutmix, MultilabelStratifiedKFold, train_model_v2_1, class balance, SENetEncoder_Multiscale_v2+ClassifierModule_v9
    calib mixup, 
    """
    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = tdu.transform_wrapper(
                            [
                            ],
                        alb.Compose([
                            #albtr.RandomGamma(p=0.8),
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                            #        albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                            #        albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                            #        ], p=0.2),
                            #alb.OneOf([
                            #        #albtr.Blur(blur_limit=4, p=0.2),
                            #        albtr.MedianBlur(blur_limit=3, p=0.2)
                            #        ], p=0.1), 
                            #alb.OneOf([
                            #        GridMask(num_grid=(3,7), mode=0, rotate=15, p=1),
                            #        GridMask(num_grid=(3,7), mode=2, rotate=15, p=1),
                            #        ], p=0.7),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )
    ts_transformer = tdu.transform_wrapper(
                        [
                        ],
                        alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )

    ## model
    #encoder = modi_senet.SENetEncoder(senet.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder_ThreeNeck(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers= [0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = 1.0
    cutmix_cand_layers = [0]
    output_layers = [2, 3, 4]
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    encoder_is_separately = three_neck
    encoder_use_mixup = True
    dropout_p = 0.0
    gem_p = None
    classifier_module = bengali_classifier.ClassifierModule_v9
    model = bengali_classifier.BengaliClassifier_v1(encoder, encoded_planes, 
                                                    encoder_is_separately, encoder_use_mixup, 
                                                    dropout_p, gem_p, 
                                                    classifier_module=classifier_module,
                                                    )

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.99, random_state=2020)
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    id = np.arange(len(labels))[:,None]

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
        if ifld == 0:
            print('training fold ', ifld)
            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            ## train with encoder freeze
            #tr_batch_size = 512
            #ts_batch_size = 512
            #tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            #vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            #
            #epochs = 2
            #lr = 1e-3 * tr_batch_size / 64
            #grad_accum_steps = 1
            #warmup_epoch=1
            #patience=5
            #factor=0.5
            #opt = 'AdaBound'
            #weight_decay=1e-4
            #loss_w = [0.5, 0.25, 0.25]
            #reference_label = labels[tr_idxs]
            #
            #model.freeze_encoder(freeze=True, target_layers=[0, 1, 2, 3, 4])
            #model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
            #                           epochs, lr, grad_accum_steps, 
            #                           warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label)
            #model.freeze_encoder(freeze=False)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 110
            lr = 1e-3
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            
            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v0_4_2():
    """
    mixup, mish, cutmix, MultilabelStratifiedKFold, train_model_v2_1, class balance(0.999), SENetEncoder_Multiscale_v2+ClassifierModule_v9
    """
    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = tdu.transform_wrapper(
                            [
                            ],
                        alb.Compose([
                            #albtr.RandomGamma(p=0.8),
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                            #        albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                            #        albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                            #        ], p=0.2),
                            #alb.OneOf([
                            #        #albtr.Blur(blur_limit=4, p=0.2),
                            #        albtr.MedianBlur(blur_limit=3, p=0.2)
                            #        ], p=0.1), 
                            #alb.OneOf([
                            #        GridMask(num_grid=(3,7), mode=0, rotate=15, p=1),
                            #        GridMask(num_grid=(3,7), mode=2, rotate=15, p=1),
                            #        ], p=0.7),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )
    ts_transformer = tdu.transform_wrapper(
                        [
                        ],
                        alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )

    ## model
    #encoder = modi_senet.SENetEncoder(senet.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder_ThreeNeck(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.2
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = 1.0
    cutmix_cand_layers = [0]
    output_layers = [2, 3, 4]
    encoder = modi_senet.SENetEncoder_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    encoder_is_separately = three_neck
    encoder_use_mixup = True
    dropout_p = 0.0
    gem_p = None
    classifier_module = bengali_classifier.ClassifierModule_v9
    model = bengali_classifier.BengaliClassifier_v1(encoder, encoded_planes, 
                                                    encoder_is_separately, encoder_use_mixup, 
                                                    dropout_p, gem_p, 
                                                    classifier_module=classifier_module,
                                                    )

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.99, random_state=2020)
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    id = np.arange(len(labels))[:,None]

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
        if ifld == 0:
            print('training fold ', ifld)
            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train with encoder freeze
            tr_batch_size = 512
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            
            epochs = 2
            lr = 1e-3 * tr_batch_size / 64
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            
            model.freeze_encoder(freeze=True, target_layers=[0, 1, 2, 3, 4])
            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta)
            model.freeze_encoder(freeze=False)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 110
            lr = 1e-3
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v0_4_2_1():
    """
    mixup, mish, cutmix, MultilabelStratifiedKFold, train_model_v2_1, class balance(0.999), SENetEncoder_Multiscale_v2+ClassifierModule_v9
    """
    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = tdu.transform_wrapper(
                            [
                            ],
                        alb.Compose([
                            #albtr.RandomGamma(p=0.8),
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                            #        albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                            #        albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                            #        ], p=0.2),
                            #alb.OneOf([
                            #        #albtr.Blur(blur_limit=4, p=0.2),
                            #        albtr.MedianBlur(blur_limit=3, p=0.2)
                            #        ], p=0.1), 
                            #alb.OneOf([
                            #        GridMask(num_grid=(3,7), mode=0, rotate=15, p=1),
                            #        GridMask(num_grid=(3,7), mode=2, rotate=15, p=1),
                            #        ], p=0.7),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )
    ts_transformer = tdu.transform_wrapper(
                        [
                        ],
                        alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )

    ## model
    #encoder = modi_senet.SENetEncoder(senet.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder_ThreeNeck(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.2
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = 1.0
    cutmix_cand_layers = [0]
    output_layers = [2, 3, 4]
    encoder = modi_senet.SENetEncoder_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    encoder_is_separately = three_neck
    encoder_use_mixup = True
    dropout_p = 0.0
    gem_p = None
    classifier_module = bengali_classifier.ClassifierModule_v9
    model = bengali_classifier.BengaliClassifier_v1(encoder, encoded_planes, 
                                                    encoder_is_separately, encoder_use_mixup, 
                                                    dropout_p, gem_p, 
                                                    classifier_module=classifier_module,
                                                    )

    model.load_state_dict(torch.load(os.path.join(pretrain_path_root, 'pretrained_200303_se_resnext50_32x4d_cb_multi2-3-4_v0', 'checkpoint'))['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.99, random_state=2020)
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    id = np.arange(len(labels))[:,None]

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
        if ifld == 0:
            print('training fold ', ifld)
            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            #epochs = 110
            #lr = 1e-3
            #grad_accum_steps = 1
            #warmup_epoch=1
            #patience=5
            #factor=0.5
            #opt = 'AdaBound'
            #weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            #reference_label = labels[tr_idxs]
            #cb_beta = 0.999

            #model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
            #                           epochs, lr, grad_accum_steps, 
            #                           warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta)

            tru.test_model_v1(model, tr_loader, vl_loader, loss_w)

            # save
            #torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v0_4_2_tta1():
    """
    mixup, mish, cutmix, MultilabelStratifiedKFold, train_model_v2_1, class balance(0.999), SENetEncoder_Multiscale_v2+ClassifierModule_v9
    """
    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            #albtr.RandomGamma(p=0.8),
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                            #        albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                            #        albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                            #        ], p=0.2),
                            #alb.OneOf([
                            #        #albtr.Blur(blur_limit=4, p=0.2),
                            #        albtr.MedianBlur(blur_limit=3, p=0.2)
                            #        ], p=0.1), 
                            #alb.OneOf([
                            #        GridMask(num_grid=(3,7), mode=0, rotate=15, p=1),
                            #        GridMask(num_grid=(3,7), mode=2, rotate=15, p=1),
                            #        ], p=0.7),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    #encoder = modi_senet.SENetEncoder(senet.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder_ThreeNeck(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.2
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = 1.0
    cutmix_cand_layers = [0]
    output_layers = [2, 3, 4]
    encoder = modi_senet.SENetEncoder_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    encoder_is_separately = three_neck
    encoder_use_mixup = True
    dropout_p = 0.0
    gem_p = None
    classifier_module = bengali_classifier.ClassifierModule_v9
    model = bengali_classifier.BengaliClassifier_v1(encoder, encoded_planes, 
                                                    encoder_is_separately, encoder_use_mixup, 
                                                    dropout_p, gem_p, 
                                                    classifier_module=classifier_module,
                                                    )

    model.load_state_dict(torch.load(os.path.join(pretrain_path_root, 'pretrained_200303_se_resnext50_32x4d_cb_multi2-3-4_v0', 'checkpoint'))['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.99, random_state=2020)
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    id = np.arange(len(labels))[:,None]

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
        if ifld == 0:
            print('training fold ', ifld)
            #vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # test
            ts_batch_size = 512
            #vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            #loss_w = [0.5, 0.25, 0.25]
            #tru.test_model_v1(model, vl_loader, vl_loader, loss_w)

            # test tta
            print('tta')
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0, 0.1, -0.1], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0, -10, 10], scales=[0], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0, -30, 30], scales=[0], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0], shifts=[[0,0], [-0.05,-0.05], [0.05, 0.05]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0, -10, 10], scales=[0, 0.1, -0.1], shifts=[[0,0], [-0.05,-0.05], [0.05, 0.05]])
            tta = pru.TTA(model, ts_transformer, degrees=[0, -10, 10], scales=[0, 0.1, -0.1], shifts=[[0,0]])
            

            #tta = pru.TTA(model, ts_transformer, degrees=[0, -15, 15], scales=[0], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0, -0.05, 0.05], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0, -0.15, 0.15], shifts=[[0,0]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0], shifts=[[0,0], [-0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [0.05, -0.05]])
            
            pred_label = pru.logit3_to_label(tta.predict(imgs[vl_idxs], ts_batch_size))

            gra_score = tru.macro_recall(labels[vl_idxs, 0], pred_label[0])
            vow_score = tru.macro_recall(labels[vl_idxs, 1], pred_label[1])
            con_score = tru.macro_recall(labels[vl_idxs, 2], pred_label[2])
            print('total_score ', 0.5 * gra_score + 0.25 * vow_score + 0.25 * con_score)
            print('gra_score ', gra_score)
            print('vow_score ', vow_score)
            print('con_score ', con_score)
            print(0.5 * gra_score + 0.25 * vow_score + 0.25 * con_score, ',', gra_score, ',', vow_score, ',', con_score)


    return


def make_model_v0_4_3():
    """
    mixup, mish, cutmix, MultilabelStratifiedKFold, train_model_v2_1, class balance(0.999), SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock
    """
    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = tdu.transform_wrapper(
                            [
                            ],
                        alb.Compose([
                            #albtr.RandomGamma(p=0.8),
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                            #        albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                            #        albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                            #        ], p=0.2),
                            #alb.OneOf([
                            #        #albtr.Blur(blur_limit=4, p=0.2),
                            #        albtr.MedianBlur(blur_limit=3, p=0.2)
                            #        ], p=0.1), 
                            #alb.OneOf([
                            #        GridMask(num_grid=(3,7), mode=0, rotate=15, p=1),
                            #        GridMask(num_grid=(3,7), mode=2, rotate=15, p=1),
                            #        ], p=0.7),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )
    ts_transformer = tdu.transform_wrapper(
                        [
                        ],
                        alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
                        )

    ## model
    #encoder = modi_senet.SENetEncoder(senet.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    #encoder = modi_senet.SENetEncoder_ThreeNeck(senet.se_resnet50(num_classes=1000, pretrained='imagenet'))
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.2
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    encoder_is_separately = three_neck
    encoder_use_mixup = True
    dropout_p = 0.0
    gem_p = None
    classifier_module = bengali_classifier.ClassifierModule_v9
    model = bengali_classifier.BengaliClassifier_v1(encoder, encoded_planes, 
                                                    encoder_is_separately, encoder_use_mixup, 
                                                    dropout_p, gem_p, 
                                                    classifier_module=classifier_module,
                                                    )

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.99, random_state=2020)
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    id = np.arange(len(labels))[:,None]

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
        if ifld == 0:
            print('training fold ', ifld)
            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            ## train with encoder freeze
            #tr_batch_size = 512
            #ts_batch_size = 512
            #tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            #vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            #
            #epochs = 2
            #lr = 1e-3 * tr_batch_size / 64
            #grad_accum_steps = 1
            #warmup_epoch=1
            #patience=5
            #factor=0.5
            #opt = 'AdaBound'
            #weight_decay=1e-4
            #loss_w = [0.5, 0.25, 0.25]
            #reference_label = labels[tr_idxs]
            #cb_beta = 0.999
            #
            #model.freeze_encoder(freeze=True, target_layers=[0, 1, 2, 3, 4])
            #model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
            #                           epochs, lr, grad_accum_steps, 
            #                           warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta)
            #model.freeze_encoder(freeze=False)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 110
            lr = 1e-3
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_0():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 1000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 120
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def test_model_v1_0_0():
    CHECKPOINT_PATH = '../trained_model/20200308_make_model_v1_0_0/checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                                    albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                                    albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                                    ], p=0.5),
                            alb.OneOf([
                                    albtr.MedianBlur(blur_limit=3, p=0.2)
                                    ], p=0.3), 
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])


    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            vl_idxs = np.arange(len(labels))
            #vl_idxs = vl_idxs[:1000]

            # test
            ts_batch_size = 512
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            loss_w = [0.5, 0.25, 0.25]
            tru.test_model_v1(model, None, vl_loader, loss_w)

            # tta
            #print('tta')
            #ts_batch_size = 512
            #tta = pru.TTA(model, ts_transformer, degrees=[0, -10, 10], scales=[0, 0.1, -0.1], shifts=[[0,0], [-0.05,-0.05], [0.05, 0.05]])
            #tta = pru.TTA(model, ts_transformer, degrees=[0], scales=[0], shifts=[[0,0]])
            #pred_logit = tta.predict(imgs[vl_idxs], ts_batch_size)
            #pred_label = pru.logit3_to_label(pred_logit)
            #
            #gra_score = tru.macro_recall(labels[vl_idxs, 0], pred_label[0])
            #vow_score = tru.macro_recall(labels[vl_idxs, 1], pred_label[1])
            #con_score = tru.macro_recall(labels[vl_idxs, 2], pred_label[2])
            #print('total_score ', 0.5 * gra_score + 0.25 * vow_score + 0.25 * con_score)
            #print('gra_score ', gra_score)
            #print('vow_score ', vow_score)
            #print('con_score ', con_score)
            #print(0.5 * gra_score + 0.25 * vow_score + 0.25 * con_score, ',', gra_score, ',', vow_score, ',', con_score)
            #
            #tru.save_preds(labels[vl_idxs,0], labels[vl_idxs,1], labels[vl_idxs,2], pred_label[0], pred_label[1], pred_label[2], pred_logit[0], pred_logit[1], pred_logit[2], 'tta_vl_')

def make_model_v1_0_0_1():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            #albtr.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 1000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 120
            lr = 5e-5
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.9999
            #start_epoch = 0 if CP is None else CP['epoch']
            #opt_state_dict = None if CP is None else CP['optimizer']
            start_epoch = 0
            opt_state_dict = None

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return


def make_model_v1_0_1():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.99), 
    """
    CHECKPOINT_PATH = 'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 60
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.99
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def test_model_v1_0_1():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.99), 
    """
    CHECKPOINT_PATH = '../trained_model/20200308_make_model_v1_0_1/checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            #albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                                    albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                                    albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                                    ], p=0.5),
                            alb.OneOf([
                                    albtr.MedianBlur(blur_limit=3, p=0.2)
                                    ], p=0.3), 
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])


    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            vl_idxs = np.arange(len(labels))

            # test
            ts_batch_size = 512

            tr_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], tr_transformer)
            tr_loader = tdu.get_dataloader(tr_ds, ts_batch_size, shuffle=False)

            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            
            loss_w = [0.5, 0.25, 0.25]
            #tru.test_model_v1(model, tr_loader, vl_loader, loss_w)
            tru.test_model_v1(model, None, vl_loader, loss_w)

    return

def make_model_v1_0_3():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.99), 
    """
    CHECKPOINT_PATH = None #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [3, 4] #[2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [1024, 2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 60
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.99
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_4():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.99), 
    """
    CHECKPOINT_PATH = None #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2, 3]
    cutmix_alpha = 1.0
    cutmix_cand_layers = [0, 1, 2, 3]
    #output_layers = [3, 4] #[2, 3, 4]
    #dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_Mixup(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            #output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = 2048 #[1024, 2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v1(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 60
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.99
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_5():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2, 3] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [4] #[2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 60
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def test_model_v1_0_5():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = '../trained_model/20200310_make_model_v1_0_5/checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            #albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                                    albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                                    albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                                    ], p=0.5),
                            alb.OneOf([
                                    albtr.MedianBlur(blur_limit=3, p=0.2)
                                    ], p=0.3), 
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2, 3] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [4] #[2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            vl_idxs = np.arange(len(labels))

            # test
            ts_batch_size = 512

            tr_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], tr_transformer)
            tr_loader = tdu.get_dataloader(tr_ds, ts_batch_size, shuffle=False)

            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            
            loss_w = [0.5, 0.25, 0.25]
            #tru.test_model_v1(model, tr_loader, vl_loader, loss_w)
            tru.test_model_v1(model, None, vl_loader, loss_w)

    return

def make_model_v1_0_5_1():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [4] #[2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 60
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 1e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']
            #start_epoch = 0
            #opt_state_dict = None

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_6():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 

    PreprocPipeline_v4
    """
    CHECKPOINT_PATH = None #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    imgs = dt_utl.get_image(type_is_train=True, height=137, width=236, data_idxs=[0,1,2,3])
    pp_pl = img_prc.PreprocPipeline_v4()
    imgs = pp_pl.preprocessing(imgs)
    pp_pl.save_imgs(imgs)
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2, 3] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [4] #[2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 60
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_7():
    """
    mish, mixup, cutmix, 
    upsampling
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.99), 
    """
    CHECKPOINT_PATH = None #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0, 1, 2, 3]
    #output_layers = [3, 4] #[2, 3, 4]
    dropblock_p = 0.2
    upsample_size = (32*6, 32*6)
    calib_mixup = True
    encoder = modi_senet.SENetEncoder_Mixup(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            dropblock_p=dropblock_p, upsample_size=upsample_size, calib_mixup=calib_mixup,
                                            )
    encoded_planes = 2048 #[1024, 2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v1(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 24
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.99
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return


def make_model_v1_0_8():
    """
    mish, mixup, cutmix, 
    upsampling
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0, 1, 2, 3]
    #output_layers = [3, 4] #[2, 3, 4]
    dropblock_p = 0.2
    upsample_size = None
    calib_mixup = True
    encoder = modi_senet.SENetEncoder_Mixup(get_senet(num_classes=1000, pretrained=None), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            dropblock_p=dropblock_p, upsample_size=upsample_size, calib_mixup=calib_mixup,
                                            )
    encoded_planes = 2048 #[1024, 2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v1(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 1e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']
            #start_epoch = 0
            #opt_state_dict = None
            
            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def test_model_v1_0_8():
    """
    mish, mixup, cutmix, 
    upsampling
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = '../trained_model/20200313_make_model_v1_0_8/checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            #albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                                    albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                                    albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                                    ], p=0.5),
                            alb.OneOf([
                                    albtr.MedianBlur(blur_limit=3, p=0.2)
                                    ], p=0.3), 
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0, 1, 2, 3]
    #output_layers = [3, 4] #[2, 3, 4]
    dropblock_p = 0.2
    upsample_size = None
    calib_mixup = True
    encoder = modi_senet.SENetEncoder_Mixup(get_senet(num_classes=1000, pretrained=None), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            dropblock_p=dropblock_p, upsample_size=upsample_size, calib_mixup=calib_mixup,
                                            )
    encoded_planes = 2048 #[1024, 2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v1(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            vl_idxs = np.arange(len(labels))

            # test
            ts_batch_size = 512

            tr_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], tr_transformer)
            tr_loader = tdu.get_dataloader(tr_ds, ts_batch_size, shuffle=False)

            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            
            loss_w = [0.5, 0.25, 0.25]
            #tru.test_model_v1(model, tr_loader, vl_loader, loss_w)
            tru.test_model_v1(model, None, vl_loader, loss_w)

    return

def make_model_v1_0_9():
    """
    mish, mixup, cutmix, 
    upsampling
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v1()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext101_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.2
    mix_cand_layers=[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0, 1, 2, 3]
    #output_layers = [3, 4] #[2, 3, 4]
    dropblock_p = 0.2
    upsample_size = None
    calib_mixup = False
    encoder = modi_senet.SENetEncoder_Mixup(get_senet(num_classes=1000, pretrained=None), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            dropblock_p=dropblock_p, upsample_size=upsample_size, calib_mixup=calib_mixup,
                                            )
    encoded_planes = 2048 #[1024, 2048] #[512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v1(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 10000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 140
            lr = 1e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']
            #start_epoch = 0
            #opt_state_dict = None
            
            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)
            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_10():
    """
    PreprocPipeline_v5
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    class balance(0.999), 
    """
    CHECKPOINT_PATH = None #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    imgs = dt_utl.get_image(type_is_train=True, height=137, width=236, data_idxs=[0,1,2,3])
    pp_pl = img_prc.PreprocPipeline_v5()
    imgs = pp_pl.preprocessing(imgs)
    pp_pl.save_imgs(imgs)
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 0.5
    mix_cand_layers=[0, 1, 2, 3] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained='imagenet'), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 1000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 120
            lr = 5e-4
            grad_accum_steps = 1
            warmup_epoch=1
            patience=5
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def make_model_v1_0_11():
    """
    PreprocPipeline_v5
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    class balance(0.999), 
    """
    CHECKPOINT_PATH = 'checkpoint' #'checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    #imgs = dt_utl.get_image(type_is_train=True, height=137, width=236, data_idxs=[0,1,2,3])
    pp_pl = img_prc.PreprocPipeline_v5()
    #imgs = pp_pl.preprocessing(imgs)
    #pp_pl.save_imgs(imgs)
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained=None), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    ## training
    #fld = ShuffleSplit(n_splits=5, test_size=.001, random_state=2020)
    #fld = MultilabelStratifiedKFold(n_splits=1000, random_state=2020)
    #id = np.arange(len(labels))[:,None]

    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(id, labels)):
    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            tr_idxs = np.arange(len(labels))
            vl_idxs = np.random.choice(tr_idxs, 1000, replace=False)

            tr_ds = tdu.ImgDataset(imgs[tr_idxs], labels[tr_idxs], tr_transformer)
            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)

            # train
            tr_batch_size = 64
            ts_batch_size = 512
            tr_loader = tdu.get_dataloader(tr_ds, tr_batch_size)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)

            epochs = 120
            lr = 1e-4
            grad_accum_steps = 1
            warmup_epoch=2
            patience=3
            factor=0.5
            opt = 'AdaBound'
            weight_decay=1e-4 #0.0
            loss_w = [0.5, 0.25, 0.25]
            reference_label = labels[tr_idxs]
            cb_beta = 0.999
            start_epoch = 0 if CP is None else CP['epoch']
            opt_state_dict = None if CP is None else CP['optimizer']
            #start_epoch = 0
            #opt_state_dict = None

            model = tru.train_model_v2_1(model, tr_loader, vl_loader, 
                                       epochs, lr, grad_accum_steps, 
                                       warmup_epoch, patience, factor, opt, weight_decay, loss_w, reference_label, cb_beta, start_epoch, opt_state_dict)

            
            # save
            torch.save(model.state_dict(), 'bengali_model')

    return

def test_model_v1_0_11():
    """
    mish, mixup, cutmix, 
    SENetEncoder_Multiscale_v2+ClassifierModule_v9, 
    dropblock,

    MultilabelStratifiedKFold, 
    class balance(0.999), 
    """
    CHECKPOINT_PATH = '../trained_model/20200316_make_model_v1_0_11/checkpoint' # None
    CP = get_checkpoint(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None

    ## data
    pp_pl = img_prc.PreprocPipeline_v5()
    imgs = pp_pl.load_imgs()

    labels = dt_utl.get_train_label()

    # transformer
    tr_transformer = alb.Compose([
                            #albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])
    ts_transformer = alb.Compose([
                            albtr.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=1),
                            alb.OneOf([albtr.ElasticTransform(p=0.1, alpha=30, sigma=7, alpha_affine=0,border_mode=cv2.BORDER_CONSTANT),
                                    #albtr.GridDistortion(num_steps=20, distort_limit=0.3 ,border_mode=cv2.BORDER_CONSTANT,p=0.1),
                                    albtr.OpticalDistortion(distort_limit=0.8, shift_limit=0.3,border_mode=cv2.BORDER_CONSTANT, p=0.1)                  
                                    ], p=0.5),
                            alb.OneOf([
                                    albtr.MedianBlur(blur_limit=3, p=0.2)
                                    ], p=0.3), 
                            albtr.Normalize(0.5, 0.5),
                            albToTensor()
                            ])

    ## model
    get_senet = senet.se_resnext50_32x4d #senet.se_resnet152, senet.se_resnext50_32x4d, senet.se_resnext101_32x4d, senet.se_resnet50
    input3ch = False
    three_neck = False
    use_mish = True
    mixup_alpha = 1.0
    mix_cand_layers=[0, 1, 2] #[0, 1, 2, 3]
    cutmix_alpha = None #1.0
    cutmix_cand_layers = None #[0]
    output_layers = [2, 3, 4]
    dropblock_p = 0.2
    encoder = modi_senet.SENetEncoder_CalibMixup_Multiscale_v2(get_senet(num_classes=1000, pretrained=None), 
                                            input3ch=input3ch,
                                            three_neck=three_neck, mixup_alpha=mixup_alpha, mix_cand_layers=mix_cand_layers,
                                            use_mish=use_mish, cutmix_alpha=cutmix_alpha, cutmix_cand_layers=cutmix_cand_layers, 
                                            output_layers=output_layers, dropblock_p=dropblock_p,
                                            )
    encoded_planes = [512, 1024, 2048] #4 * 512

    dropout_p = 0.1
    classifier = bengali_classifier.ClassifierModule_v9(encoded_planes, dropout_p=dropout_p)

    encoder_use_mixup = True
    model = bengali_classifier.BengaliClassifier_v2(encoder, classifier, 
                                                    encoder_use_mixup, 
                                                    )
    if CP is not None:
        model.load_state_dict(CP['state_dict'])

    for ifld in range(1):
        if ifld == 0:
            print('training fold ', ifld)
            vl_idxs = np.arange(len(labels))

            # test
            ts_batch_size = 512

            tr_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], tr_transformer)
            tr_loader = tdu.get_dataloader(tr_ds, ts_batch_size, shuffle=False)

            vl_ds = tdu.ImgDataset(imgs[vl_idxs], labels[vl_idxs], ts_transformer)
            vl_loader = tdu.get_dataloader(vl_ds, ts_batch_size, shuffle=False)
            
            loss_w = [0.5, 0.25, 0.25]
            #tru.test_model_v1(model, tr_loader, vl_loader, loss_w)
            tru.test_model_v1(model, None, vl_loader, loss_w)

    return