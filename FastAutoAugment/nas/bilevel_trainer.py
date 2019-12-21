from typing import Optional
import os
from FastAutoAugment.nas.model_desc import ModelDesc

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from overrides import overrides

from ..common.trainer import Trainer
from .bilevel_optimizer import BilevelOptimizer
from ..nas.model import Model
from ..nas.vis_genotype import draw_model_desc
from ..common.metrics import Metrics


class BilevelTrainer(Trainer):
    def __init__(self, w_momentum:float, w_decay:float, alpha_optim:Optimizer,
                 max_final_edges:int, plotsdir:str,
                 model:Model, lossfn: _Loss,
                 aux_weight: float, grad_clip: float,
                 drop_path_prob: float, logger_freq: int,
                 tb_tag: str, val_logger_freq:int, val_tb_tag:str)->None:
        super().__init__(model, lossfn, lossfn,
                         aux_weight=aux_weight, grad_clip=grad_clip,
                         drop_path_prob=drop_path_prob,
                         logger_freq=logger_freq, tb_tag=tb_tag)

        self.w_momentum, self.w_decay = w_momentum, w_decay
        self.alpha_optim = alpha_optim
        self.max_final_edges, self.plotsdir = max_final_edges, plotsdir
        self._bilevel_optim = BilevelOptimizer(w_momentum, w_decay, alpha_optim,
                                     True, model, lossfn)
        self.best_model_desc = self._get_model_desc()

    @overrides
    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        super().pre_epoch(train_dl, val_dl, train_metrics,  val_metrics)

        # prep val set to train alphas
        assert val_dl is not None
        self._valid_iter = iter(val_dl)

    @overrides
    def pre_step(self, x:Tensor, y:Tensor, train_metrics:Metrics)->None:
        super().pre_step(x, y, train_metrics)

        # reset val loader if we exausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device, non_blocking=True)

        # update alphas
        self._bilevel_optim.step(x, y, x_val, y_val, train_metrics.get_cur_lr(), self.optim)

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
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






