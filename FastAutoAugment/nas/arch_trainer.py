from typing import Optional, Tuple, Union
import os
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides, EnforceOverrides

from ..common.config import Config
from ..common import utils
from ..common import common
from ..nas.model import Model
from ..common.metrics import Metrics
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_model_desc import draw_model_desc


class ArchTrainer(Trainer, EnforceOverrides):
    def __init__(self, conf_train:Config, model:Model, device)->None:
        super().__init__(conf_train, model, device)

        self._max_final_edges = conf_train['max_final_edges']
        self._plotsdir = common.get_logdir(conf_train['plotsdir'], True)

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        del self._valid_iter # clean up
        super().post_epoch(train_dl, val_dl)

        self._update_best()

    def _update_best(self)->None:
        train_metrics, val_metrics = self.get_metrics()
        if (val_metrics and val_metrics.is_best()) or \
                (train_metrics and train_metrics.is_best()):

            # log model_desc as a image
            plot_filepath = os.path.join(self._plotsdir, "EP{train_metrics.epoch:03d}")
            draw_model_desc(self.get_model_desc(), plot_filepath+"-normal",
                            caption=f"Epoch {train_metrics.epoch}")

    def get_model_desc(self)->ModelDesc:
        return self.model.finalize(max_edges=self._max_final_edges)
