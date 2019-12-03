from typing import Tuple
import  torch.nn as nn
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
import  torchvision.datasets as tvds
from  torch.utils.data import DataLoader
import numpy as np
import os

from ..common.config import Config
from .arch_cnn_model import ArchCnnModel
from .arch import Arch
from ..common.data import get_dataloaders
from ..common.common import get_logger, get_tb_writer
from ..common import utils
from ..common.optimizer import get_lr_scheduler, get_optimizer
from .vis_genotype import draw_genotype

def search_arch(conf:Config)->None:
    logger = get_logger()

    # region conf vars
    conf_search   = conf['darts']['search']
    conf_loader   = conf['darts']['search']['loader']
    conf_ds       = conf['dataset']
    conf_w_opt    = conf_search['weights']['optimizer']
    conf_w_sched  = conf_search['weights']['lr_schedule']
    ds_name       = conf_ds['name']
    dataroot      = conf['dataroot']
    aug           = conf_loader['aug']
    cutout        = conf_loader['cutout']
    val_ratio     = conf_loader['val_ratio']
    batch_size    = conf_loader['batch']
    epochs        = conf_loader['epochs']
    val_fold      = conf_loader['val_fold']
    horovod       = conf['horovod']
    ch_in         = conf_ds['ch_in']
    ch_out_init   = conf_search['ch_out_init']
    n_classes     = conf_ds['n_classes']
    n_layers      = conf_search['layers']
    report_freq   = conf['report_freq']
    grad_clip     = conf_w_opt['clip']
    plotsdir      = conf['plotsdir']
    logdir        = conf['logdir']
    # endregion

    # breakdown train to train + val split
    train_dl, val_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=True, load_test=False,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod)

    # CIFAR classification task
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    model = ArchCnnModel(ch_in, ch_out_init, n_classes, n_layers,
                        criterion).to(device)
    logger.info("Total param size = %f MB", utils.count_parameters_in_MB(model))

    # trainer for alphas
    arch = Arch(conf, model)

    # optimizer for w
    w_optim = get_optimizer(conf_w_opt, model.weights())
    lr_scheduler = get_lr_scheduler(conf_w_sched, epochs, w_optim)

    # in search phase we typically only run 50 epochs
    best_top1, best_genotype = 0., None
    for epoch in range(epochs):
        lr = lr_scheduler.get_lr()[0]

        logger.info('\nEpoch: %d lr: %e', epoch, lr)
        model.print_alphas(logger)

        global_step = epoch * len(train_dl) # for plotting graphs

        # training
        _train_epoch(train_dl, val_dl, model, arch, w_optim, lr,
            device, grad_clip, report_freq,
            epoch, epochs, global_step)

        # validation
        val_top1, _ = _validate_epoch(val_dl, model, device,
            report_freq, epoch, epochs, global_step)

        lr_scheduler.step()

        # log results of this epoch
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        # genotype as a image
        plot_filepath = os.path.join(plotsdir, "EP{:03d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        draw_genotype(genotype.normal, plot_filepath+"-normal", caption=caption)
        draw_genotype(genotype.reduce, plot_filepath+"-reduce", caption=caption)

        # save
        if best_top1 < val_top1:
            best_top1, best_genotype, is_best = val_top1, genotype, True
        else:
            is_best = False
        utils.save_checkpoint(model, is_best, logdir)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def _train_epoch(train_dl:DataLoader, val_dl:DataLoader, model:ArchCnnModel,
    arch:Arch, w_optim:Optimizer, lr:float, device, grad_clip:float,
    report_freq, epoch:int, epochs:int, global_step:int) \
        ->Tuple[float, float]:

    logger, writer = get_logger(), get_tb_writer()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    train_size = len(train_dl)
    writer.add_scalar('train/lr', lr, global_step)

    valid_iter = iter(val_dl) # we will reset iterator if |val| < |train|

    for step, (x_train, y_train) in enumerate(train_dl):
        # we may be switching to eval during arch step
        model.train()

        # one blocking and another non blocking so we don't keep going
        x_train, y_train = \
            x_train.to(device), y_train.to(device, non_blocking=True)

        # reset val loader if we exausted it
        try:
            x_val, y_val = next(valid_iter)
        except StopIteration:
            # reinit iterator
            valid_iter = iter(val_dl)
            x_val, y_val = next(valid_iter)
        x_val, y_val = x_val.to(device), y_val.to(device, non_blocking=True)

        # update alphas
        arch.step(x_train, y_train, x_val, y_val, lr, w_optim)
        # update weights
        w_optim.zero_grad()
        logits = model(x_train)
        loss = model.criterion(logits, y_train)
        loss.backward()
        # TODO: original darts clips alphas as well
        nn.utils.clip_grad_norm_(model.weights(), grad_clip)
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

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(
                    epoch+1, epochs, top1.avg))

    return top1.avg, losses.avg


def _validate_epoch(val_dl:DataLoader, model:ArchCnnModel, device,
    report_freq:int, epoch:int, epochs:int, global_step:int)->Tuple[float, float]:
    """ Evaluate model on validation set """

    logger, writer = get_logger(), get_tb_writer()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    val_size = len(val_dl)

    model.eval()
    with torch.no_grad():
        for step, (x_val, y_val) in enumerate(val_dl):
            # one blocking, another non-blocking so we don't keep going
            x_val, y_val = x_val.to(device), y_val.cuda(non_blocking=True)
            batch_size = x_val.size(0)

            logits = model(x_val)
            loss = model.criterion(logits, y_val)

            prec1, prec5 = utils.accuracy(logits, y_val, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            if step % report_freq == 0 or step == val_size-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}"
                    " Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, epochs, step, val_size-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, global_step)
    writer.add_scalar('val/top1', top1.avg, global_step)
    writer.add_scalar('val/top5', top5.avg, global_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(
                        epoch+1, epochs, top1.avg))

    return top1.avg, losses.avg