import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler

from adabound import AdaBound
from model import mixup
from model import class_balanced_loss as cbl

from tqdm import tqdm
import numpy as np
import pandas as pd
import sklearn.metrics

import warnings
warnings.simplefilter('ignore')

# utils
def save_log(loglist, filename, use_orth_loss=False, use_confusion=False):
    df = pd.DataFrame(loglist)
    if use_orth_loss:
        df.columns = ['epoch', 
                'tr_total_loss', 'tr_gra_loss', 'tr_vow_loss', 'tr_con_loss', 
                'tr_score', 'tr_gra_score', 'tr_vow_score', 'tr_con_score', 
                'tr_orthogonal_loss',
                'vl_total_loss', 'vl_gra_loss', 'vl_vow_loss', 'vl_con_loss', 
                'vl_score', 'vl_gra_score', 'vl_vow_score', 'vl_con_score',
                ]
    elif use_confusion:
        df.columns = ['epoch', 
                      'tr_total_loss', 'tr_gra_loss', 'tr_vow_loss', 'tr_con_loss', 'tr_confusion_loss', 
                      'tr_score', 'tr_gra_score', 'tr_vow_score', 'tr_con_score',
                      'vl_total_loss', 'vl_gra_loss', 'vl_vow_loss', 'vl_con_loss' , 'vl_confusion_loss', 
                      'vl_score', 'vl_gra_score', 'vl_vow_score', 'vl_con_score',
                      ]
    else:
        df.columns = ['epoch', 
                      'tr_total_loss', 'tr_gra_loss', 'tr_vow_loss', 'tr_con_loss', 
                      'tr_score', 'tr_gra_score', 'tr_vow_score', 'tr_con_score',
                      'vl_total_loss', 'vl_gra_loss', 'vl_vow_loss', 'vl_con_loss', 
                      'vl_score', 'vl_gra_score', 'vl_vow_score', 'vl_con_score',
                      ]
    df.to_csv(filename)
    return

def save_checkpoint(epoch, model, optimizer, file_name):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), }
    torch.save(state, file_name)
    return

# scheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

# loss
def multiloss_wrapper_v1(loss_funcs, weights, loss_funcs_use_all=None):
    def lossfunc(outputs, targets):
        total_loss = 0
        losses = []
        for i in range(len(loss_funcs)):
            loss = loss_funcs[i](outputs[i], targets[:,i])
            total_loss += weights[i] * loss
            losses.append(loss.item())

        if loss_funcs_use_all is not None:
            for i in range(len(loss_funcs_use_all)):
                loss = loss_funcs_use_all[i](outputs, targets)
                total_loss += weights[i + len(loss_funcs)] * loss
                losses.append(loss.item())

        return total_loss, losses
    return lossfunc

def multiloss_wrapper_v1_mixup(loss_funcs, weights, loss_funcs_use_all=None):
    def lossfunc(outputs, targets_a, targets_b, mix_rate):
        total_loss = 0
        losses = []

        for i in range(len(loss_funcs)):
            loss = loss_funcs[i](outputs[i], targets_a[:,i], targets_b[:,i], mix_rate)
            total_loss += weights[i] * loss
            losses.append(loss.item())

        if loss_funcs_use_all is not None:
            for i in range(len(loss_funcs_use_all)):
                loss = loss_funcs_use_all[i](outputs, targets_a, targets_b, mix_rate)
                total_loss += weights[i + len(loss_funcs)] * loss
                losses.append(loss.item())

        return total_loss, losses
    return lossfunc

# metric
def macro_recall(true, pred):
    mr = sklearn.metrics.recall_score(true, pred, average='macro')
    return mr

# train script
def _trainer_v1(net, loader, criterion, optimizer, 
             now_epoch, grad_accum_steps, warmup_epoch, warmup_scheduler, use_mixup=False):
    net.train()
    total_loss = 0
    gra_loss = 0
    vow_loss = 0
    con_loss = 0
    other_loss = 0

    total_num = 0
    gra_pred = []
    vow_pred = []
    con_pred = []
    gra_true = []
    vow_true = []
    con_true = []

    optimizer.zero_grad()
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
        if now_epoch < warmup_epoch:
            warmup_scheduler.step()

        imgs = imgs.cuda()
        labels = labels.cuda()

        if use_mixup:
            outputs, label_a, label_b, mix_rate = net(imgs, labels)
            loss, losses = criterion(outputs, label_a, label_b, mix_rate)
        else:
            outputs = net(imgs)
            loss, losses = criterion(outputs, labels)

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # loss
        total_loss += loss.item() * grad_accum_steps
        gra_loss += losses[0]
        vow_loss += losses[1]
        con_loss += losses[2]
        if len(losses) > 3:
            other_loss += losses[3]
                
        # score
        with torch.no_grad():
            gra_pred.append(outputs[0].max(1)[1].cpu().numpy())
            vow_pred.append(outputs[1].max(1)[1].cpu().numpy())
            con_pred.append(outputs[2].max(1)[1].cpu().numpy())
            gra_true.append(labels[:,0].cpu().numpy())
            vow_true.append(labels[:,1].cpu().numpy())
            con_true.append(labels[:,2].cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)
    gra_loss = gra_loss / (batch_idx + 1)
    vow_loss = vow_loss / (batch_idx + 1)
    con_loss = con_loss / (batch_idx + 1)
    if len(losses) > 3:
        other_loss = other_loss / (batch_idx + 1)

    # score
    gra_score = macro_recall(np.concatenate(gra_true), np.concatenate(gra_pred))
    vow_score = macro_recall(np.concatenate(vow_true), np.concatenate(vow_pred))
    con_score = macro_recall(np.concatenate(con_true), np.concatenate(con_pred))
    score = 0.5 * gra_score + 0.25 * vow_score + 0.25 * con_score

    print('Train Loss: %.3f | Macro Recall: %.3f' % (total_loss, score))
    print('Gra. Loss: %.3f | Macro Recall: %.3f' % (gra_loss, gra_score))
    print('Vow. Loss: %.3f | Macro Recall: %.3f' % (vow_loss, vow_score))
    print('Con. Loss: %.3f | Macro Recall: %.3f' % (con_loss, con_score))
    if len(losses) > 3:
        print('Oth. Loss: %.3f | Macro Recall: ---' % (other_loss))
    
    return now_epoch, total_loss, gra_loss, vow_loss, con_loss, score, gra_score, vow_score, con_score

def _tester_v1(net, loader, criterion, loss_scale=None, save_result=None):
    loss_s = 1.0 if loss_scale is None else loss_scale
    net.eval()
    total_loss = 0
    gra_loss = 0
    vow_loss = 0
    con_loss = 0

    total_num = 0
    gra_pred = []
    vow_pred = []
    con_pred = []
    gra_logit = []
    vow_logit = []
    con_logit = []
    gra_true = []
    vow_true = []
    con_true = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
            imgs = imgs.cuda()
            labels = labels.cuda()

            outputs = net(imgs)
            loss, losses = criterion(outputs, labels)
            loss = loss * loss_s

            # loss
            total_loss += loss.item()
            gra_loss += losses[0]
            vow_loss += losses[1]
            con_loss += losses[2]

            # score
            gra_pred.append(outputs[0].max(1)[1].cpu().numpy())
            vow_pred.append(outputs[1].max(1)[1].cpu().numpy())
            con_pred.append(outputs[2].max(1)[1].cpu().numpy())
            gra_true.append(labels[:,0].cpu().numpy())
            vow_true.append(labels[:,1].cpu().numpy())
            con_true.append(labels[:,2].cpu().numpy())

            if save_result is not None:
                gra_logit.append(outputs[0].cpu().numpy())
                vow_logit.append(outputs[1].cpu().numpy())
                con_logit.append(outputs[2].cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)
    gra_loss = gra_loss / (batch_idx + 1)
    vow_loss = vow_loss / (batch_idx + 1)
    con_loss = con_loss / (batch_idx + 1)

    gra_true = np.concatenate(gra_true)
    gra_pred = np.concatenate(gra_pred)
    vow_true = np.concatenate(vow_true)
    vow_pred = np.concatenate(vow_pred)
    con_true = np.concatenate(con_true)
    con_pred = np.concatenate(con_pred)
    if save_result is not None:
        gra_logit = np.concatenate(gra_logit)
        vow_logit = np.concatenate(vow_logit)
        con_logit = np.concatenate(con_logit)

    # score
    gra_score = macro_recall(gra_true, gra_pred)
    vow_score = macro_recall(vow_true, vow_pred)
    con_score = macro_recall(con_true, con_pred)
    score = 0.5 * gra_score + 0.25 * vow_score + 0.25 * con_score

    print('Test Loss: %.3f | Macro Recall: %.3f' % (total_loss, score))
    print('Gra. Loss: %.3f | Macro Recall: %.3f' % (gra_loss, gra_score))
    print('Vow. Loss: %.3f | Macro Recall: %.3f' % (vow_loss, vow_score))
    print('Con. Loss: %.3f | Macro Recall: %.3f' % (con_loss, con_score))
    
    if save_result is not None:
        save_preds(gra_true, vow_true, con_true, gra_pred, vow_pred, con_pred, gra_logit, vow_logit, con_logit, save_result)

    return total_loss, gra_loss, vow_loss, con_loss, score, gra_score, vow_score, con_score

def save_preds(gra_true, vow_true, con_true, gra_pred, vow_pred, con_pred, gra_logit, vow_logit, con_logit, name):
    df = np.concatenate([gra_true[:,None], gra_pred[:,None], gra_logit], axis=1)
    df = pd.DataFrame(df)
    df.to_csv(name + 'gra_pred.csv')

    df = np.concatenate([vow_true[:,None], vow_pred[:,None], vow_logit], axis=1)
    df = pd.DataFrame(df)
    df.to_csv(name + 'vow_pred.csv')

    df = np.concatenate([con_true[:,None], con_pred[:,None], con_logit], axis=1)
    df = pd.DataFrame(df)
    df.to_csv(name + 'con_pred.csv')
    return


# train model
def train_model_v2_1(net, trainloader, validloader, 
             epochs, lr, grad_accum_steps=1, 
             warmup_epoch=1, patience=5, factor=0.5, opt='AdaBound', weight_decay=0.0, loss_w=[0.5, 0.25, 0.25], reference_labels=None, cb_beta=0.99, 
             start_epoch=0, opt_state_dict=None):
    """
    mixup, ReduceLROnPlateau, class balance
    """
    net = net.cuda()

    # loss
    loss_w = loss_w if loss_w is not None else [0.5, 0.25, 0.25]
    if reference_labels is None:
        if len(loss_w) == 3:
            criterion = multiloss_wrapper_v1_mixup(loss_funcs=[mixup.CrossEntropyLossForMixup(num_class=168), 
                                                               mixup.CrossEntropyLossForMixup(num_class=11), 
                                                               mixup.CrossEntropyLossForMixup(num_class=7)],
                                                    weights=loss_w)
        elif len(loss_w) == 4:
            criterion = multiloss_wrapper_v1_mixup(loss_funcs=[mixup.CrossEntropyLossForMixup(num_class=168), 
                                                               mixup.CrossEntropyLossForMixup(num_class=11), 
                                                               mixup.CrossEntropyLossForMixup(num_class=7),
                                                               mixup.CrossEntropyLossForMixup(num_class=1292)],
                                                    weights=loss_w)

    else:
        if len(loss_w) == 3:
            criterion = multiloss_wrapper_v1_mixup(loss_funcs=[cbl.CB_CrossEntropyLoss(reference_labels[:,0], num_class=168, beta=cb_beta, label_smooth=0.0), 
                                                               cbl.CB_CrossEntropyLoss(reference_labels[:,1], num_class=11, beta=cb_beta, label_smooth=0.0), 
                                                               cbl.CB_CrossEntropyLoss(reference_labels[:,2], num_class=7, beta=cb_beta, label_smooth=0.0)],
                                                    weights=loss_w)
        elif len(loss_w) == 4:
            criterion = multiloss_wrapper_v1_mixup(loss_funcs=[cbl.CB_CrossEntropyLoss(reference_labels[:,0], num_class=168, beta=cb_beta, label_smooth=0.0), 
                                                               cbl.CB_CrossEntropyLoss(reference_labels[:,1], num_class=11, beta=cb_beta, label_smooth=0.0), 
                                                               cbl.CB_CrossEntropyLoss(reference_labels[:,2], num_class=7, beta=cb_beta, label_smooth=0.0),
                                                               cbl.CB_CrossEntropyLoss(reference_labels[:,3], num_class=1292, beta=cb_beta, label_smooth=0.0)],
                                                    weights=loss_w)

    test_criterion = multiloss_wrapper_v1(loss_funcs=[nn.CrossEntropyLoss(), 
                                                      nn.CrossEntropyLoss(), 
                                                      nn.CrossEntropyLoss()],
                                            weights=loss_w)

    # opt
    if opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif opt == 'AdaBound':
        optimizer = AdaBound(net.parameters(), lr=lr, final_lr=0.1, weight_decay=weight_decay)

    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor, verbose=True)
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    if opt_state_dict is not None:
        optimizer.load_state_dict(opt_state_dict)
    
    # train
    loglist = []
    val_loss = 100
    for epoch in range(start_epoch, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(val_loss)

        print('epoch ', epoch)
        tr_log = _trainer_v1(net, trainloader, criterion, optimizer, epoch, grad_accum_steps, warmup_epoch, warmup_scheduler, use_mixup=True)
        vl_log = _tester_v1(net, validloader, test_criterion)
        loglist.append(list(tr_log) + list(vl_log))

        val_loss = vl_log[0]

        save_checkpoint(epoch, net, optimizer, 'checkpoint')
        save_log(loglist, 'training_log.csv')

    return net

def test_model_v1(net, trainloader, validloader, loss_w=[0.5, 0.25, 0.25]):
    net = net.cuda()

    loss_w = loss_w if loss_w is not None else [0.5, 0.25, 0.25]
    test_criterion = multiloss_wrapper_v1(loss_funcs=[nn.CrossEntropyLoss(), 
                                                      nn.CrossEntropyLoss(), 
                                                      nn.CrossEntropyLoss()],
                                            weights=loss_w)

    if trainloader is not None:
        print('train data')
        tr_log = _tester_v1(net, trainloader, test_criterion, save_result='tr_')
    if validloader is not None:
        print('test data')
        vl_log = _tester_v1(net, validloader, test_criterion, save_result='vl_')

    return