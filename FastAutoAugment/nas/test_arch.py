from typing import Optional
from FastAutoAugment.common.utils import save
import os

import torch
import torch.nn as nn
import yaml

from ..common import utils
from ..common.trainer import Trainer
from ..common.config import Config
from ..common.data import get_dataloaders
from .model import Model
from .model_desc import ModelDesc, RunMode
from .model_desc_builder import ModelDescBuilder

def test_arch(conf_common:Config, conf_data:Config, conf_test:Config,
              template_model_desc:Optional[ModelDesc]=None,
              save_model:bool=True):

    # region conf vars
    logger_freq       = conf_common['logger_freq']
    horovod           = conf_common['horovod']
    logdir            = conf_common['logdir']
    # dataset
    ds_name           = conf_data['name']
    max_batches       = conf_data['max_batches']
    dataroot          = conf_data['dataroot']
    # dataloader
    conf_loader       = conf_test['loader']
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
    model_desc_file = conf_test['model_desc_file']
    model_file    = conf_test['model_file']
    conf_model_desc   = conf_test['model_desc']
    aux_weight        = conf_model_desc['aux_weight']
    drop_path_prob    = conf_model_desc['drop_path_prob']
    # optimizer
    conf_opt          = conf_test['optimizer']
    grad_clip         = conf_opt['clip']
    conf_lr_sched     = conf_test['lr_schedule']
    # endregion

    if logdir and not template_model_desc:
        # open the model description we want to test
        with open(os.path.join(logdir, model_desc_file), 'r') as f:
            template_model_desc = yaml.load(f, Loader=yaml.Loader)

    # template desc must be supplied either via parameter of config
    assert template_model_desc is not None
    # if save_model is true then logdir must be specified
    assert (save_model and logdir) or not save_model

    # Use template_model_desc to create new model description that has
    # structure as specified in config
    builder = ModelDescBuilder(conf_data, conf_model_desc,
                               run_mode=RunMode.EvalTrain,
                               template=template_model_desc)
    model_desc = builder.get_model_desc()

    # get data
    train_dl, _, test_dl, _ = get_dataloaders(
        ds_name, batch_size, dataroot, aug, cutout,
        load_train=True, load_test=True,
        val_ratio=0., val_fold=0, # no validation set
        horovod=horovod, max_batches=max_batches, n_workers=n_workers)
    assert train_dl is not None and test_dl is not None

    device = torch.device("cuda")

    train_lossfn = utils.get_lossfn(conf_train_lossfn).to(device)
    test_lossfn = utils.get_lossfn(conf_test_lossfn).to(device)

    # create model
    model = Model(model_desc)
    if data_parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    optim = utils.get_optimizer(conf_opt, model.parameters())
    lr_scheduler = utils.get_lr_scheduler(conf_lr_sched, epochs, optim)

    trainer = Trainer(model, device, train_lossfn, test_lossfn,
        aux_weight=aux_weight, grad_clip=grad_clip,
        drop_path_prob=drop_path_prob, logger_freq=logger_freq,
        title='eval_train', val_logger_freq=1000, val_title='eval_test')
    train_metrics, test_metrics = trainer.fit(train_dl, test_dl, epochs,
                                              optim, lr_scheduler)
    test_metrics.report_best()

    if save_model:
        utils.save(model, os.path.join(logdir, model_file))

    return test_metrics.best_top1, model







