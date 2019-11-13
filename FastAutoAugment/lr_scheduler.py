import torch

from theconf import Config as C


def adjust_learning_rate_resnet(optimizer):
    """
    Sets the learning rate to the initial LR decayed by 10 on every predefined epochs
    Ref: AutoAugment
    """

    if C.get()['epoch'] == 90:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
    elif C.get()['epoch'] == 270:   # autoaugment
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])
    else:
        raise ValueError('invalid epoch=%d for resnet scheduler' % C.get()['epoch'])
