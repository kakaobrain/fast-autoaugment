from typing import Tuple
import  torch.nn as nn
import torch
from torch import optim
import  torchvision.datasets as tvds
from  torch.utils.data import DataLoader
import numpy as np
import os

from ..common.config import Config
from .arch_cnn_model import ArchCnnModel
from .arch import Arch
from ..common.data import get_dataloaders
from ..common.common import get_logger, create_tb_writers
from ..common import utils
from ..common.optimizer import get_lr_scheduler, get_optimizer
from .vis_genotype import draw_genotype

def search_arch(conf:Config)->None:
    logger = get_logger()

    if not conf['darts']['bilevel']:
        logger.warn('bilevel arg is NOT true. This is useful only for abalation study for bilevel optimization!')

    device = torch.device('cuda')
    writer = create_tb_writers(conf)[0]

    # breakdown train to train + val split, usually 50-50%
    _, train_dl, val_dl, _ = get_dataloaders(conf['dataset'], conf['batch'],
        conf['dataroot'], conf['aug'], conf['darts']['search_cutout'],
        val_ratio=conf['val_ratio'], val_fold=conf['val_fold'], horovod=conf['horovod'])

    # CIFAR classification task
    criterion = nn.CrossEntropyLoss().to(device)

    model = ArchCnnModel(conf['ch_in'], conf['darts']['ch_out_init'], conf['n_classes'], conf['darts']['layers'], criterion).to(device)
    logger.info("Total param size = %f MB", utils.count_parameters_in_MB(model))

    # trainer for alphas
    arch = Arch(model, conf)

    # optimizer for the model weight
    w_optim = get_optimizer(conf['optimizer'], model.parameters())
    lr_scheduler = get_lr_scheduler(conf, w_optim)

    # in search phase we typically only run 50 epochs
    best_top1, best_genotype = 0., None
    for epoch in range(conf['epochs']):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        logger.info('\nEpoch: %d lr: %e', epoch, lr)
        model.print_alphas(logger)

        global_step = epoch * len(train_dl)

        # training
        _train_epoch(train_dl, val_dl, model, arch, criterion, w_optim, lr,
            device, conf['optimizer']['clip'], conf['report_freq'],
            epoch, conf['epochs'], global_step, writer)

        # validation
        val_top1 = _validate_epoch(val_dl, model, criterion, device, conf['report_freq'],
            epoch, conf['epochs'], global_step, writer)

        genotype = model.genotype()
        logger.info('Genotype: %s', genotype)

        # save
        if best_top1 < val_top1:
            best_top1 = val_top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, is_best, conf['logdir'])

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def _train_epoch(train_dl:DataLoader, val_dl:DataLoader, model:ArchCnnModel, arch:Arch, criterion, w_optim, lr:float,
    device, grad_clip:float, report_freq, epoch:int, epochs:int, global_step:int, writer)->Tuple[float, float]:

    logger = get_logger()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    train_size = len(train_dl)
    writer.add_scalar('train/lr', lr, global_step)

    valid_iter = iter(val_dl)

    for step, (x_train, y_train) in enumerate(train_dl):
        model.train() # make sure model is in train mode

        # [b, 3, 32, 32], [40]
        # one blocking and another non blocking
        x_train, y_train = x_train.to(device), y_train.to(device, non_blocking=True)

        try:
            x_val, y_val = next(valid_iter) # [b, 3, 32, 32], [b]
        except StopIteration:
            # reinit iterator
            valid_iter = iter(val_dl)
            x_val, y_val = next(valid_iter) # [b, 3,

        x_val, y_val = x_val.to(device), y_val.to(device, non_blocking=True)

        # 1. update alpha
        arch.step(x_train, y_train, x_val, y_val, lr, w_optim)

        w_optim.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()

        # clip all w grades (original darts clips alphas as well)
        nn.utils.clip_grad_norm_(model.weights(), grad_clip)
        # as our arch parameters (i.e. alpha) is kept seperate, they don't get updated
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, y_train, topk=(1, 5))
        losses.update(loss.item(), x_train.size(0))
        top1.update(prec1.item(), x_train.size(0))
        top5.update(prec5.item(), x_train.size(0))

        if step % report_freq == 0 or step == train_size-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, epochs, step, train_size-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/top1', prec1.item(), global_step)
        writer.add_scalar('train/top5', prec5.item(), global_step)
        global_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, epochs, top1.avg))

    return top1.avg, losses.avg


def _validate_epoch(val_dl, model, criterion, device, report_freq,
    epoch, epochs, global_step, writer)->Tuple[float, float]:
    """ Evaluate model on validation set """

    logger = get_logger()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    val_size = len(val_dl)

    model.eval()

    with torch.no_grad():
        for step, (x_val, y_val) in enumerate(val_dl):
            # one blocking, another non-blocking
            x_val, y_val = x_val.to(device), y_val.cuda(non_blocking=True)
            batch_size = x_val.size(0)

            logits = model(x_val)
            loss = criterion(logits, y_val)

            prec1, prec5 = utils.accuracy(logits, y_val, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            if step % report_freq == 0 or step == val_size-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, epochs, step, val_size-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, global_step)
    writer.add_scalar('val/top1', top1.avg, global_step)
    writer.add_scalar('val/top5', top5.avg, global_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, epochs, top1.avg))

    return top1.avg, losses.avg