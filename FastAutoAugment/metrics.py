import copy

import torch
import numpy as np
from collections import defaultdict

from torch import nn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res


def cross_entropy_smooth(input, target, size_average=True, label_smoothing=0.1):
    y = torch.eye(10).cuda()
    lb_oh = y[target]

    target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


import time
import torch

from theconf import Config as C
from FastAutoAugment.archive import policy_decoder
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.networks import num_class, get_model


def eval_tta(config, augment, reporter):
    C.get()
    C.get().conf = config

    cv_ratio_test, cv_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']

    # setup - provided augmentation rules
    if 'aug' in augment:
        C.get()['aug'] = augment['aug']
    else:
        C.get()['aug'] = policy_decoder(augment, augment['num_policy'], augment['num_op'])

    # eval
    model = get_model(C.get()['model'], num_class(C.get()['dataset']), data_parallel=True)
    ckpt = torch.load(save_path)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    loaders = []
    for _ in range(augment['num_policy']):  # TODO
        _, tl, validloader, tl2 = get_dataloaders(C.get()['dataset'], C.get()['batch'], augment['dataroot'], cv_ratio_test, split_idx=cv_fold, target_lb=augment.get('class_idx', -1))
        loaders.append(iter(validloader))
        del tl, tl2

    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)
                data = data.cuda()
                label = label.cuda()

                pred = model(data)
                loss = loss_fn(pred, label)
                losses.append(loss.detach().cpu().numpy())

                _, pred = pred.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                corrects.append(correct)
                del loss, correct, pred, data, label

            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()

            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict({
                'minus_loss': -1 * np.sum(losses_min),
                'correct': np.sum(corrects_max),
                'cnt': len(corrects_max)
            })
            del corrects, corrects_max
    except StopIteration:
        pass

    del model
    metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    if reporter is not None:
        reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'], elapsed_time=gpu_secs, done=True)
    return metrics['correct']
