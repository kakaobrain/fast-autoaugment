from typing import Callable, Tuple, Optional

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .tester import Tester
from .config import Config
from . import utils
from ..common.common import get_logger

class Trainer(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module, device)->None:
        # region config vars
        conf_lossfn = conf_train['lossfn']
        self._aux_weight = conf_train['aux_weight']
        self._grad_clip = conf_train['grad_clip']
        self._drop_path_prob = conf_train['drop_path_prob']
        self._logger_freq = conf_train['logger_freq']
        self._title = conf_train['title']
        self._epochs = conf_train['epochs']
        self._conf_optim = conf_train['optimizer']
        self._conf_sched = conf_train['lr_schedule']
        conf_validation = conf_train['validation']
        # endregion

        self.model = model
        self.device = device
        self._lossfn = utils.get_lossfn(conf_lossfn).to(device)
        self._metrics = self._create_metrics(self._epochs, None)
        self._tester = Tester(conf_validation, model, device) \
                        if conf_validation else None

    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        optim = self.get_optimizer()
        lr_scheduler = self.get_scheduler(optim)
        self._metrics = self._create_metrics(self._epochs, optim)

        self.pre_fit(train_dl, val_dl, optim, lr_scheduler)
        for epoch in range(self._epochs):
            self._set_drop_path(epoch, self._epochs)

            self.pre_epoch(train_dl, val_dl)
            self._train_epoch(train_dl, optim)
            self.post_epoch(train_dl, val_dl)

            lr_scheduler.step()
        self.post_fit(train_dl, val_dl)

    def get_optimizer(self)->Optimizer:
        return utils.get_optimizer(self._conf_optim, self.model.parameters())

    def get_scheduler(self, optim:Optimizer)->_LRScheduler:
        return utils.get_lr_scheduler(self._conf_sched, self._epochs, optim)

    def get_metrics(self)->Tuple[Metrics, Optional[Metrics]]:
        return self._metrics, self._tester.get_metrics() if self._tester else None

    def get_cur_lr(self)->float:
        return self._metrics.get_cur_lr()

    def pre_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                optim:Optimizer, sched:_LRScheduler)->None:
        pass
    def post_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        pass
    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        self._metrics.pre_epoch()
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        self._metrics.post_epoch()
        if val_dl and self._tester:
            self._tester.test(val_dl)
    def pre_step(self, x:Tensor, y:Tensor, optim:Optimizer)->None:
        self._metrics.pre_step(x, y)
    def post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int)->None:
        self._metrics.post_step(x, y, logits, loss, steps)

    def _create_metrics(self, epochs:int, optim:Optional[Optimizer]):
        return Metrics(self._title, epochs,
                       optim=optim, logger_freq=self._logger_freq)

    def _train_epoch(self, train_dl: DataLoader, optim:Optimizer)->None:
        steps = len(train_dl)
        self.model.train()
        for x, y in train_dl:
            assert self.model.training # derived class might alter the mode

            # enable non-blocking on 2nd part so its ready when we get to it
            x, y = x.to(self.device), y.to(self.device, non_blocking=True)

            self.pre_step(x, y, optim)

            optim.zero_grad()

            if self._aux_weight > 0.0:
                logits, aux_logits = self.model(x)
            else:
                (logits, *_), aux_logits = self.model(x), None
            loss = self.compute_loss(self._lossfn, x, y, logits,
                                    self._aux_weight, aux_logits)

            loss.backward()

            if self._grad_clip:
                # TODO: original darts clips alphas as well but pt.darts doesn't
                nn.utils.clip_grad_norm_(self.model.weights(), self._grad_clip)
            optim.step()

            self.post_step(x, y, logits, loss, steps)

    def compute_loss(self, lossfn:Callable,
                     x:Tensor, y:Tensor, logits:Tensor,
                     aux_weight:float, aux_logits:Optional[Tensor])->Tensor:
        logger = get_logger()
        loss = lossfn(logits, y)
        if aux_weight > 0.0:
            if aux_logits is not None:
                loss += aux_weight * lossfn(aux_logits, y)
            else:
                logger.warn(f'aux_weight is {aux_weight} but aux tower was not generated')
        return loss

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self._drop_path_prob:
            drop_prob = self._drop_path_prob * epoch / epochs
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
                                       .format(self._drop_path_prob))
