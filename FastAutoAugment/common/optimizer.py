from torch.optim import lr_scheduler, SGD, Adam, Optimizer
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import torch
from torch import nn

def get_optimizer(conf:dict, params)->Optimizer:
    if conf['type'] == 'sgd':
        return SGD(
           params,
            lr=conf['lr'],
            momentum=conf['momentum'],
            weight_decay=conf['decay'],
            nesterov=conf['nesterov']
        )
    elif conf['type'] == 'adam':
         return Adam(params,
            lr=conf['lr'],
            betas=conf['betas'],
            weight_decay=conf['decay'])
    else:
        raise ValueError('invalid optimizer type=%s' % conf['type'])

def get_optim_lr(optimizer:Optimizer)->float:
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_scheduler(conf_lrs:dict, epochs:int, optimizer:Optimizer)-> \
        _LRScheduler:

    scheduler:_LRScheduler = None
    lr_scheduler_type = conf_lrs['type'] # TODO: default should be none?
    if lr_scheduler_type == 'cosine':
        # adjust max epochs for warmup
        # TODO: shouldn't we be increasing epochs or schedule lr only after warmup?
        if conf_lrs.get('warmup', None):
            epochs -= conf_lrs['warmup']['epoch']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
            eta_min=conf_lrs['lr_min'])
    elif lr_scheduler_type == 'resnet':
        scheduler = _adjust_learning_rate_resnet(optimizer, epochs)
    elif lr_scheduler_type == 'pyramid':
        scheduler = _adjust_learning_rate_pyramid(optimizer, epochs,
            get_optim_lr(optimizer))
    elif lr_scheduler_type == 'step':
        decay_period = conf_lrs['decay_period']
        gamma = conf_lrs['gamma']
        scheduler = lr_scheduler.StepLR(optimizer, decay_period, gamma=gamma)
    elif not lr_scheduler_type:
            scheduler = None # TODO: check support for this or use StepLR
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    # select warmup for LR schedule
    if conf_lrs.get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=conf_lrs['warmup']['multiplier'],
            total_epoch=conf_lrs['warmup']['epoch'],
            after_scheduler=scheduler
        )
    return scheduler

def _adjust_learning_rate_pyramid(optimizer, max_epoch:int, base_lr:float):
    def _internal_adjust_learning_rate_pyramid(epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = base_lr * (0.1 ** (epoch // (max_epoch * 0.5))) * (0.1 ** (epoch // (max_epoch * 0.75)))
        return lr    lossfn =

    return lr_scheduler.LambdaLR(optimizer, _internal_adjust_learning_rate_pyramid)


def _adjust_learning_rate_resnet(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every [90, 180, 240] epochs
    Ref: AutoAugment
    """

    if epoch == 90:
        return lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
    elif epoch == 270:   # autoaugment
        return lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])
    else:
        raise ValueError('invalid epoch=%d for resnet scheduler' % epoch)

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

def get_lossfn(conf_lossfn:dict, conf_dataset:dict)->nn._Loss:
    type = conf_lossfn['type']
    if type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif type == 'CrossEntropyLabelSmooth':
        return SmoothCrossEntropyLoss(smoothing=conf_lossfn['smoothing'])
    else:
        raise ValueError('criterian type "{}" is not supported'.format(type))