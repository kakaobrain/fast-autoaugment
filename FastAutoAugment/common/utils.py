import  os
from typing import Iterable
import  numpy as np
import  torch
import  shutil
from torch import nn
from torch.optim.optimizer import Optimizer
import  torchvision.transforms as transforms


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def param_size(model):
    """count all parameters excluding auxiliary"""
    return np.sum(v.numel() for name, v in model.named_parameters() \
        if "auxiliary" not in name) / 1e6


def save_checkpoint(model:nn.Module, optim:Optimizer, best_top1:float,
        epoch:int, is_best:bool, ckpt_dir:str)->None:
    state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_top1': best_top1,
            'optim': optim.state_dict(),
    }
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    print('saved to model:', model_path)
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    print('load from model:', model_path)
    model.load_state_dict(torch.load(model_path))


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # Bernoulli returns 1 with pobability p and 0 with 1-p.
        # Below generates tensor of shape (batch,1,1,1) filled with 1s and 0s
        #   as per keep_prob.
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob) \
            .to(device=x.device)
        # scale tensor by 1/p as we will be losing other values
        # for each tensor in batch, zero out values with probability p
        x.div_(keep_prob).mul_(mask)
    return x

def first_or_default(it:Iterable, default=None):
    for i in it:
        return i
    return default