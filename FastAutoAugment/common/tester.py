from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics

class Tester(EnforceOverrides):
    """Evaluate model on given data"""

    def __init__(self, model:nn.Module, device, lossfn:_Loss,
                 logger_freq:int=10, tb_tag:str='')->None:
        self.model = model
        self.device = device
        self.lossfn = lossfn
        self.tb_tag = tb_tag
        self.logger_freq = logger_freq

    def test(self, test_dl:DataLoader, epochs:int)->Metrics:
        metrics = self.create_metrics(epochs)
        for epoch in range(epochs):
            self.test_epoch(test_dl, metrics)
        return metrics

    def pre_epoch(self, epoch_steps:int, metrics:Metrics)->None:
        metrics.pre_epoch()
    def post_epoch(self, epoch_steps:int, metrics:Metrics)->None:
        metrics.post_epoch()
    def pre_step(self, x:Tensor, y:Tensor, metrics:Metrics)->None:
        metrics.pre_step(x, y)
    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int, metrics:Metrics)->None:
        metrics.post_step(x, y, logits, loss, steps)

    def create_metrics(self, epochs:int):
        return Metrics(epochs, self.tb_tag, self.logger_freq)

    def test_epoch(self, test_dl: DataLoader, metrics:Metrics)->None:
        steps = len(test_dl)
        self.pre_epoch(steps, metrics)

        self.model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                assert not self.model.training # derived class might alter the mode

                # enable non-blocking on 2nd part so its ready when we get to it
                x, y = x.to(self.device), y.to(self.device, non_blocking=True)

                self.pre_step(x, y, metrics)
                logits, *_ = self.model(x) # ignore aux logits in test mode
                loss = self.lossfn(logits, y)
                self.post_step(x, y, logits, loss, steps, metrics)

        self.post_epoch(steps, metrics)

