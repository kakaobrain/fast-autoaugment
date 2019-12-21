import os

import torch
import torch.nn as nn

from ..common import utils
from ..common.common import get_logger, get_tb_writer
from FastAutoAugment.common.trainer import Trainer
from FastAutoAugment.common.config import Config
from ..common.data import get_dataloaders
from .model import Model
from ..common.optimizer import get_lr_scheduler, get_optimizer, get_lossfn
from .model_desc import ModelDesc

def test_arch(conf_common:Config, conf_data:Config, conf_test:Config,
              model_desc:ModelDesc, save_model:bool=True):

    logger, writer = get_logger(), get_tb_writer()

    # region conf vars
    chkptdir          = conf_common['chkptdir']
    report_freq       = conf_common['report_freq']
    horovod           = conf_common['horovod']
    logdir            = conf_common['logdir']
    # dataset
    ds_name           = conf_data['name']
    max_batches       = conf_data['max_batches']
    dataroot          = conf_data['dataroot']
    # dataloader
    conf_loader       = conf_test['loader']
    cutout            = conf_loader['cutout']
    aug               = conf_loader['aug']
    cutout            = conf_loader['cutout']
    batch_size        = conf_loader['batch']
    epochs            = conf_loader['epochs']
    n_workers         = conf_loader['n_workers']
    data_parallel     = conf_test['data_parallel']
    # loss functions
    conf_train_lossfn = conf_test['train_lossfn']
    conf_test_lossfn  = conf_test['test_lossfn']
    # test model
    model_save_file    = conf_test['model_save_file']
    conf_model_desc   = conf_test['model_desc']
    aux_weight        = conf_model_desc['aux_weight']
    drop_path_prob    = conf_model_desc['drop_path_prob']
    # optimizer
    conf_opt          = conf_test['optimizer']
    grad_clip         = conf_opt['clip']
    conf_lr_sched     = conf_test['lr_schedule']
    # endregion

    # get data
    train_dl, _, test_dl, _ = get_dataloaders(
        ds_name, batch_size, dataroot, aug, cutout,
        load_train=True, load_test=True,
        val_ratio=0., val_fold=0, # no validation set
        horovod=horovod, max_batches=max_batches, n_workers=n_workers)

    device = torch.device("cuda")

    train_lossfn = get_lossfn(conf_train_lossfn, conf_data).to(device)
    test_lossfn = get_lossfn(conf_test_lossfn, conf_data).to(device)

    # create model
    model = Model(model_desc)
    if data_parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    optim = get_optimizer(conf_opt, model.parameters())
    lr_scheduler = get_lr_scheduler(conf_lr_sched, epochs, optim)

    trainer = Trainer(model, train_lossfn, aux_weight, grad_clip, drop_path_prob,
        report_freq, tb_tag='eval_train',
        val_logger_freq=1000, val_tb_tag='eval_test')
    train_metrics, test_metrics = trainer.fit(train_dl, test_dl, epochs,
                                              optim, lr_scheduler)
    test_metrics.report_best()

    if save_model:
        utils.save(model, os.path.join(logdir, model_save_file))

    return test_metrics.best_top1, model







