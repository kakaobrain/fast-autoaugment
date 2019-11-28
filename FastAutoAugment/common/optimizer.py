from torch.optim import lr_scheduler, SGD, Adam, Optimizer
from warmup_scheduler import GradualWarmupScheduler
from .common import Config

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

def get_scheduler(conf:Config, optimizer:Optimizer)->lr_scheduler._LRScheduler:
    lr_scheduler_type = conf['lr_schedule']['type'] # TODO: default should be none?
    scheduler = None
    if lr_scheduler_type == 'cosine':
        t_max = conf['epoch']
        # adjust max epochs for warmup
        # TODO: shouldn't we be increasing t_max or schedule lr only after warmup?
        if conf['lr_schedule'].get('warmup', None):
            t_max -= conf['lr_schedule']['warmup']['epoch']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=conf['lr_schedule']['lr_min'])
    elif lr_scheduler_type == 'resnet':
        scheduler = _adjust_learning_rate_resnet(optimizer, conf['epoch'])
    elif lr_scheduler_type == 'pyramid':
        scheduler = _adjust_learning_rate_pyramid(optimizer, conf['epoch'], conf['optimizer']['lr'])
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)
    # select warmup for LR schedule
    if conf['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=conf['lr_schedule']['warmup']['multiplier'],
            total_epoch=conf['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )
    return scheduler

def _adjust_learning_rate_pyramid(optimizer, max_epoch:int, base_lr:float):
    def _internal_adjust_learning_rate_pyramid(epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = base_lr * (0.1 ** (epoch // (max_epoch * 0.5))) * (0.1 ** (epoch // (max_epoch * 0.75)))
        return lr

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
