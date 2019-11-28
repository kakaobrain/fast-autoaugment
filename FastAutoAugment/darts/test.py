import os,sys,glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tvds
import torch.backends.cudnn as cudnn

from . import genotypes
from ..common import utils
from ..common.common import get_logger
from .model_test import NetworkCIFAR as Network

def test_arch(conf):
    logger = get_logger()

    # equal to: genotype = genotypes.DARTS_v2
    genotype = eval("genotypes.%s" % conf['darts']['test_genotype'])
    print('Load genotype:', genotype)
    model = Network(conf['darts']['init_ch'], conf['num_classes'], conf['darts']['layers'],
        conf['darts']['test_auxtowers'], genotype).cuda()

    model_filepath = os.path.join(conf['logdir'], conf['darts']['test_model_filename'])
    utils.load(model, model_filepath)
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()

    _, test_transform = utils._data_transforms_cifar10(conf['cutout'])
    test_data = tvds.CIFAR10(root=conf['dataroot'], train=False, download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=conf['batch'], shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = conf['drop_path_prob']
    test_acc, test_obj = _infer(test_queue, model, criterion, conf['report_freq'])
    logger.info('test_acc %f', test_acc)


def _infer(test_queue, model, criterion, report_freq):
    logger = get_logger()

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (x, target) in enumerate(test_queue):
            x, target = x.cuda(), target.cuda(non_blocking=True)

            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            batchsz = x.size(0)
            objs.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % report_freq == 0:
                logger.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

