import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as tvds
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

from ..common import utils
from ..common.common import get_logger, get_tb_writer, test_epoch, train_epoch
from ..common.data import get_dataloaders
from .model_test import NetworkCIFAR as Network
from ..common.optimizer import get_lr_scheduler, get_optimizer

def test_arch(conf):
    logger, writer = get_logger(), get_tb_writer()

    # region conf vars
    conf_ds       = conf['dataset']
    dataroot      = conf['dataroot']
    conf_test   = conf['darts']['test']
    conf_loader   = conf_test['loader']
    cutout        = conf_loader['cutout']
    test_genotype     = conf_test['test_genotype']
    ch_out_init   = conf_test['ch_out_init']
    n_layers      = conf_test['layers']
    aux_weight = conf_test['aux_weight']
    drop_path_prob = conf_test['drop_path_prob']
    ds_name       = conf_ds['name']
    ch_in         = conf_ds['ch_in']
    n_classes     = conf_ds['n_classes']
    aug           = conf_loader['aug']
    cutout        = conf_loader['cutout']
    val_ratio     = conf_loader['val_ratio']
    batch_size    = conf_loader['batch']
    epochs        = conf_loader['epochs']
    conf_opt    = conf_test['optimizer']
    conf_lr_sched    = conf_test['lr_schedule']
    report_freq   = conf['report_freq']
    horovod       = conf['horovod']
    aux_weight = conf_test['aux_weight']

    # endregion

    device = torch.device("cuda")

    train_dl, _, test_dl, _ = get_dataloaders(
        ds_name, batch_size, dataroot, aug, cutout,
        load_train=True, load_test=True,
        val_ratio=0., val_fold=0, # no validation set
        horovod=horovod)

    # load genotype we want to test
    genotype = eval("genotypes.%s" % test_genotype)
    logger.info('test genotype: {}'.format(genotype))

    criterion = nn.CrossEntropyLoss().to(device)

    # create model
    model = Network(ch_out_init, n_classes, n_layers, aux_weight, genotype)
    logger.info("Model size = {:.3f} MB".format(utils.param_size(model)))
    # TODO: model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    model = model.to(device)

    optim = get_optimizer(conf_opt, model.parameters())
    lr_scheduler = get_lr_scheduler(conf_lr_sched, epochs, optim)

    test_acc, test_loss = _infer(test_dl, model, criterion, report_freq)
    logger.info('test_acc %f', test_acc)




