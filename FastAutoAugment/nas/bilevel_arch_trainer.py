from typing import Optional, Tuple

from torch.utils.data import DataLoader

from overrides import overrides

from ..common.config import Config
from ..nas.arch_trainer import ArchTrainer
from ..nas.bilevel_trainer import BilevelTrainer
from ..common.config import Config
from ..common.optimizer import get_lr_scheduler, get_optimizer, get_lossfn
from ..nas.model import Model
from ..common.metrics import Metrics
from ..nas.model_desc import ModelDesc

class BilevelArchTrainer(ArchTrainer):
    @overrides
    def fit(self, conf_search:Config, model:Model, device,
            train_dl:DataLoader, val_dl:Optional[DataLoader], epochs:int,
            plotsdir:str, logger_freq:int)->Tuple[ModelDesc, Metrics, Optional[Metrics]]:
        # region conf vars
        # search
        bilevel       = conf_search['bilevel']
        conf_lossfn   = conf_search['lossfn']
        max_final_edges = conf_search['max_final_edges']
        # optimizers
        conf_w_opt    = conf_search['weights']['optimizer']
        w_momentum    = conf_w_opt['momentum']
        w_decay       = conf_w_opt['decay']
        grad_clip     = conf_w_opt['clip']
        conf_w_sched  = conf_search['weights']['lr_schedule']
        conf_a_opt    = conf_search['alphas']['optimizer']
        # endregion

        lossfn = get_lossfn(conf_lossfn).to(device)

        # optimizer for w and alphas
        w_optim = get_optimizer(conf_w_opt, model.weights())
        alpha_optim = get_optimizer(conf_a_opt, model.alphas())
        lr_scheduler = get_lr_scheduler(conf_w_sched, epochs, w_optim)

        trainer = BilevelTrainer(w_momentum, w_decay, alpha_optim,
            max_final_edges, plotsdir, model, device, lossfn, lossfn,
            aux_weight=0.0, grad_clip=grad_clip, drop_path_prob=0.0,
            logger_freq=logger_freq, tb_tag='search_train',
            val_logger_freq=1000, val_tb_tag='search_val')

        train_metrics, val_metrics = trainer.fit(train_dl, val_dl, epochs,
                                                 w_optim, lr_scheduler)

        val_metrics.report_best() if val_metrics else train_metrics.report_best()

        return trainer.best_model_desc, train_metrics, val_metrics