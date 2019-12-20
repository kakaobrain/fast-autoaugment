from typing import Tuple, Any, Iterable, Optional
import  torch.nn as nn
import torch
import os
import yaml

from ..common.config import Config
from .model import Model
from .arch import Arch
from ..common.data import get_dataloaders
from ..common.common import get_logger, get_tb_writer
from ..common.optimizer import get_lr_scheduler, get_optimizer, get_lossfn
from .vis_genotype import draw_model_desc
from ..common.train_test_utils import train_test
from .model_desc import ModelDesc, RunMode
from .model_desc_builder import ModelDescBuilder
from .strategy import Strategy

def search_arch(conf_common:Config, conf_data:Config, conf_search:Config,
                strategy:Strategy)->ModelDesc:
    logger = get_logger()

    # region conf vars
    horovod       = conf_common['horovod']
    report_freq   = conf_common['report_freq']
    plotsdir      = conf_common['plotsdir']
    chkptdir      = conf_common['chkptdir']
    # dataset
    ds_name       = conf_data['name']
    max_batches   = conf_data['max_batches']
    dataroot      = conf_data['dataroot']
    # data loader
    conf_loader   = conf_search['loader']
    aug           = conf_loader['aug']
    cutout        = conf_loader['cutout']
    val_ratio     = conf_loader['val_ratio']
    batch_size    = conf_loader['batch']
    epochs        = conf_loader['epochs']
    val_fold      = conf_loader['val_fold']
    n_workers     = conf_loader['n_workers']
     # search
    conf_lossfn   = conf_search['lossfn']
    data_parallel = conf_search['data_parallel']
    bilevel       = conf_search['bilevel']
    conf_model_desc = conf_search['model_desc']
    max_final_edges = conf_search['max_final_edges']
    # optimizers
    conf_w_opt    = conf_search['weights']['optimizer']
    w_momentum    = conf_w_opt['momentum']
    w_decay       = conf_w_opt['decay']
    grad_clip     = conf_w_opt['clip']
    conf_w_sched  = conf_search['weights']['lr_schedule']
    conf_a_opt    = conf_search['alphas']['optimizer']
    # endregion

    builder = ModelDescBuilder(conf_data, conf_model_desc, run_mode=RunMode.Search)
    model_desc = builder.get_model_desc()
    strategy.apply(model_desc)

    # get data
    train_dl, val_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=True, load_test=False,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod,
        max_batches=max_batches, n_workers=n_workers)

    device = torch.device('cuda')

    lossfn = get_lossfn(conf_lossfn).to(device)

    model = Model(model_desc)
    if data_parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # optimizer for w and alphas
    w_optim = get_optimizer(conf_w_opt, model.weights())
    alpha_optim = get_optimizer(conf_a_opt, model.alphas())
    lr_scheduler = get_lr_scheduler(conf_w_sched, epochs, w_optim)

    # trainer for alphas
    arch = Arch(w_momentum, w_decay, alpha_optim, bilevel, model, lossfn)

    # in search phase we typically only run 50 epochs
    best_model_desc:Optional[ModelDesc] = None
    valid_iter:Optional[Iterable[Any]] = None
    def _pre_epoch(*_):
        nonlocal valid_iter
        valid_iter = iter(val_dl)

    def _post_epochfn(epoch, best_top1, top1, is_best):
        nonlocal best_model_desc

        # log results of this epoch
        model_desc = model.finalize(max_edges=max_final_edges)
        logger.info("model_desc = {}".format(model_desc))
        # model_desc as a image
        plot_filepath = os.path.join(plotsdir, "EP{:03d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        draw_model_desc(model_desc, plot_filepath+"-normal", caption=caption)

        if is_best or best_model_desc is None:
            best_model_desc = model_desc

    def _pre_stepfn(step, x_train, y_train, cur_lr):
        nonlocal valid_iter
        # reset val loader if we exausted it
        try:
            x_val, y_val = next(valid_iter)
        except StopIteration:
            # reinit iterator
            valid_iter = iter(val_dl)
            x_val, y_val = next(valid_iter)
        x_val, y_val = x_val.to(device), y_val.to(device, non_blocking=True)

        # update alphas
        arch.step(x_train, y_train, x_val, y_val, cur_lr, w_optim)

    train_test(train_dl, val_dl, model, device, lossfn, lossfn,
        w_optim, aux_weight=0.0, grad_clip=grad_clip, lr_scheduler=lr_scheduler,
        drop_path_prob=0.0, model_save_dir=chkptdir, report_freq=report_freq,
        epochs=epochs, pre_stepfn=_pre_stepfn, pre_epochfn=_pre_epoch,
        post_epochfn=_post_epochfn)

    logger.info("Best architecture\n{}".format(yaml.dump(best_model_desc)))

    assert best_model_desc is not None
    return best_model_desc


