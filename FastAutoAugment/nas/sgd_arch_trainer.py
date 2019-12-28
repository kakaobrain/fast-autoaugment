from typing import Optional, Tuple, Union
import os
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides

from ..common.config import Config
from ..nas.arch_trainer import ArchTrainer, ArchOptimizer
from ..common import utils
from ..nas.model import Model
from ..common.metrics import Metrics
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_model_desc import draw_model_desc


class SgdArchTrainer(ArchTrainer):
    def __init__(self, conf_common:Config, conf_search:Config,
                 model:Model, device)-> None:
        # region conf vars
        # common
        self.logger_freq = conf_common['logger_freq']
        self.plotsdir = conf_common['plotsdir']
        # search
        conf_lossfn   = conf_search['lossfn']
        self.max_final_edges = conf_search['max_final_edges']
        # optimizers
        self.conf_w_opt    = conf_search['optimizer']
        self.conf_a_opt    = conf_search['alphas']['optimizer']
        self.w_momentum    = self.conf_w_opt['momentum']
        self.w_decay       = self.conf_w_opt['decay']
        self.grad_clip     = self.conf_w_opt['clip']
        # endregion

        self.model = model
        self.device = device
        self.conf_search = conf_search
        self.lossfn = utils.get_lossfn(conf_lossfn).to(device)

    def get_optimizer(self)->Optimizer:
        conf_opt = self.conf_search['optimizer']
        return utils.get_optimizer(conf_opt, self.model.parameters())

    def get_scheduler(self, epochs:int, optim:Optimizer)->_LRScheduler:
        conf_sched  = self.conf_search['lr_schedule']
        return utils.get_lr_scheduler(conf_sched, epochs, optim)

    def get_sgd_trainer(self)->'SgdTrainer':
        trainer = SgdTrainer(self.model, self.device,
            self.lossfn, self.lossfn,
            aux_weight=0.0, grad_clip=self.grad_clip, drop_path_prob=0.0,
            logger_freq=self.logger_freq, title='search_train',
            val_logger_freq=1000, val_title='search_val',
            arch_optim=None, max_final_edges=self.max_final_edges, plotsdir=self.plotsdir)
        return trainer

    @overrides
    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader], epochs:int)\
            ->Tuple[ModelDesc, Metrics, Optional[Metrics]]:
        optim = self.get_optimizer()
        lr_scheduler = self.get_scheduler(epochs, optim)
        trainer = self.get_trainer()

        train_metrics, val_metrics = trainer.fit(train_dl, val_dl, epochs,
                                                 optim, lr_scheduler)

        val_metrics.report_best() if val_metrics else train_metrics.report_best()

        return trainer.best_model_desc, train_metrics, val_metrics


class SgdTrainer(Trainer):
    def __init__(self, model:Model, device, train_lossfn: _Loss, test_lossfn: _Loss,
                 aux_weight: float, grad_clip: float,
                 drop_path_prob: float, logger_freq: int,
                 title:str, val_logger_freq:int, val_title:str,
                 arch_optim:Optional[ArchOptimizer],
                 max_final_edges:int, plotsdir:str)->None:
        super().__init__(model, device, train_lossfn, test_lossfn,
                         aux_weight=aux_weight, grad_clip=grad_clip,
                         drop_path_prob=drop_path_prob, logger_freq=logger_freq, title=title, val_logger_freq=val_logger_freq, val_title=val_title)

        self._arch_optim = arch_optim
        self.max_final_edges, self.plotsdir = max_final_edges, plotsdir
        self.best_model_desc = self._get_model_desc()

    @overrides
    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        super().pre_epoch(train_dl, val_dl, train_metrics,  val_metrics)

        # prep val set to train alphas
        self._valid_iter = iter(val_dl) if val_dl is not None else None

    @overrides
    def pre_step(self, x:Tensor, y:Tensor, optim:Optimizer, train_metrics:Metrics)->None:
        super().pre_step(x, y, optim, train_metrics)

        if self._arch_optim is not None:
            if self._valid_iter is not None:
                # reset val loader if we exausted it
                try:
                    x_val, y_val = next(self._valid_iter)
                except StopIteration:
                    # reinit iterator
                    self._valid_iter = iter(self._val_dl)
                    x_val, y_val = next(self._valid_iter)
                x_val, y_val = x_val.to(self.device), y_val.to(self.device, non_blocking=True)
            else:
                x_val, y_val = None, None

            # update alphas
            self._arch_optim.step(x, y, x_val, y_val, optim, train_metrics)

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        if self._valid_iter is not None:
            del self._valid_iter # clean up
        super().post_epoch(train_dl, val_dl, train_metrics, val_metrics)

        self._update_best(train_metrics, val_metrics)

    def _update_best(self, train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        if (val_metrics and val_metrics.is_best()) or \
                (not val_metrics and train_metrics.is_best()):
            self.best_model_desc = self._get_model_desc()

            # log model_desc as a image
            plot_filepath = os.path.join(self.plotsdir, "EP{train_metrics.epoch:03d}")
            draw_model_desc(self.best_model_desc, plot_filepath+"-normal",
                            caption=f"Epoch {train_metrics.epoch}")

    def _get_model_desc(self)->ModelDesc:
        return self.model.finalize(max_edges=self.max_final_edges)
