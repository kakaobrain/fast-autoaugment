import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tvds

from ..common import utils
from ..common.common import get_logger, get_tb_writer
from ..common.data import get_dataloaders
from .model_test import NetworkCIFAR as Network

def test_arch(conf):
    logger, writer = get_logger(), get_tb_writer()
    device = torch.device("cuda")

    _, _, test_dl, *_ = get_dataloaders(
        conf['dataset'], conf['batch'], conf['dataroot'], conf['aug'],
        conf['cutout'], load_train=False, load_test=True,
        val_ratio=conf['val_ratio'], val_fold=conf['val_fold'],
        horovod=conf['horovod'])

    # equal to: genotype = genotypes.DARTS_v2
    genotype = eval("genotypes.%s" % conf['darts']['test_genotype'])
    logger.info('test genotype: {}'.format(genotype))

    model = Network(conf['darts']['ch_out_init'], conf['n_classes'],
        conf['darts']['layers'], conf['darts']['test_auxtowers'],
        genotype).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model.drop_path_prob = conf['drop_path_prob']

    model_filepath = os.path.join(conf['logdir'], conf['darts']['test_model_filename'])
    utils.load(model, model_filepath)
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

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

