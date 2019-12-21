from typing import Tuple, Optional
from abc import ABC, abstractmethod

from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .tester import Tester

class Trainer(EnforceOverrides):
    def __init__(self, model:nn.Module,
                 train_lossfn:_Loss, test_lossfn:_Loss,
                 aux_weight=0., grad_clip=0., drop_path_prob=0.,
                 logger_freq=10, tb_tag='',
                 val_logger_freq=10, val_tb_tag='')->None:
        self.model = model
        self.train_lossfn = train_lossfn
        self.aux_weight, self.grad_clip = aux_weight, grad_clip
        self.drop_path_prob = drop_path_prob
        self.tb_tag, self.logger_freq = tb_tag, logger_freq
        self.tester = Tester(model, test_lossfn, val_logger_freq, val_tb_tag)

    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader], epochs:int,
            optim:Optimizer, lr_scheduler:_LRScheduler)\
              ->Tuple[Metrics, Optional[Metrics]]:
        train_metrics, val_metrics = self.create_metrics(epochs, optim), None

        if val_dl:
            val_metrics = self.tester.create_metrics(epochs)

        for epoch in range(epochs):
            self._set_drop_path(epoch, epochs)

            self.pre_epoch(train_dl, val_dl, train_metrics, val_metrics)
            self.train_epoch(train_dl, optim, train_metrics)
            self.post_epoch(train_dl, val_dl, train_metrics, val_metrics)

            lr_scheduler.step()

        return train_metrics, val_metrics

    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        train_metrics.pre_epoch()
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                   train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        train_metrics.post_epoch()
        self.test_epoch(val_dl, val_metrics)
    def pre_step(self, x:Tensor, y:Tensor, train_metrics:Metrics)->None:
        train_metrics.pre_step(x, y)
    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int, train_metrics:Metrics)->None:
        train_metrics.post_step(x, y, logits, loss, steps)

    def create_metrics(self, epochs:int, optim:Optimizer):
        return Metrics(epochs, self.tb_tag, optim, self.logger_freq)

    def train_epoch(self, train_dl: DataLoader, optim:Optimizer,
                    train_metrics:Metrics)->None:
        steps = len(train_dl)
        self.model.train()
        for x, y in train_dl:
            assert self.model.training # derived class might alter the mode

            # enable non-blocking on 2nd part so its ready when we get to it
            x, y = x.to(self.device), y.to(self.device, non_blocking=True)

            self.pre_step(x, y, train_metrics)

            optim.zero_grad()
            if self.aux_weight > 0.:
                logits, aux_logits = self.model(x)
                loss = self.train_lossfn(logits, y)
                loss += self.aux_weight * self.train_lossfn(aux_logits, y)
            else:
                logits, *_ = self.model(x)
                loss = self.train_lossfn(logits, y)

            self.post_step(x, y, logits, loss, steps, train_metrics)

    def test_epoch(self, val_dl:Optional[DataLoader], val_metrics:Optional[Metrics]):
        if val_dl:
            self.tester.test_epoch(val_dl, val_metrics)

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self.drop_path_prob:
            drop_prob = self.drop_path_prob * epoch / epochs
            # set value as property in model (it will be used by forward())
            # this is necessory when using DataParallel(model)
            # https://github.com/pytorch/pytorch/issues/16885
            m = self.model
            if hasattr(self.model, 'module'): # for data parallel model
                m = self.model.module
            if hasattr(m, 'drop_path_prob'):
                m.drop_path_prob(drop_prob)
            else:
                raise RuntimeError('Drop path value {} was specified but model'
                                   ' does not have drop_path_prob() method'\
                                       .format(self.drop_path_prob))
