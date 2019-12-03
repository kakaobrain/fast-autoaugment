import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tvds

from ..common import utils
from ..common.common import get_logger, get_tb_writer
from ..common.data import get_dataloaders
from .model_test import NetworkCIFAR as Network
from ..common.optimizer import get_lr_scheduler, get_optimizer

def test_arch(conf):
    logger, writer = get_logger(), get_tb_writer()

    # region conf vars
    conf_test   = conf['darts']['test']
    conf_loader   = conf['darts']['test']['loader']
    conf_ds       = conf['dataset']
    conf_opt    = conf_test['optimizer']
    conf_lr_sched = conf_test['lr_schedule']
    ds_name       = conf_ds['name']
    batch_size    = conf_loader['batch']
    dataroot      = conf['dataroot']
    aug           = conf_loader['aug']
    cutout        = conf_loader['cutout']
    test_genotype     = conf_test['test_genotype']
    ch_out_init   = conf_test['ch_out_init']
    use_auxtowers = conf_test['auxtowers']
    drop_path_prob = conf_test['drop_path_prob']
    n_classes     = conf_ds['n_classes']
    # endregion

    device = torch.device("cuda")

    train_dl, _, test_dl, _ = get_dataloaders(
        ds_name, batch_size, dataroot, aug, cutout,
        load_train=True, load_test=True,
        val_ratio=0., val_fold=0, # no validation set
        horovod=conf['horovod'])

    # load genotype we want to test
    genotype = eval("genotypes.%s" % test_genotype)
    logger.info('test genotype: {}'.format(genotype))

    criterion = nn.CrossEntropyLoss().to(device)

    # create model
    model = Network(ch_out_init, n_classes, n_layers, use_auxtowers, genotype)
    model.drop_path_prob = drop_path_prob
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    # TODO: model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    model = model.to(device)

    optim = get_optimizer(conf_opt, model.parameters())
    lr_scheduler = get_lr_scheduler(conf_lr_sched, optim)

    test_acc, test_obj = _infer(test_dl, model, criterion, conf['report_freq'])
    logger.info('test_acc %f', test_acc)

def _infer(test_dl, model, criterion, report_freq):
    logger = get_logger()

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (x, target) in enumerate(test_dl):
            x, target = x.cuda(), target.cuda(non_blocking=True)

            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            batchsz = x.size(0)
            objs.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % report_freq == 0:
                logger.info('test %03d %e %f %f', step, objs.avg, top1.avg,
                    top5.avg)

    return top1.avg, objs.avg

