import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .common import get_logger, get_tb_writer
from . import utils

def test_epoch(test_dl:DataLoader, model:nn.Module, device, lossfn:_Loss,
        report_freq:int, epoch:int, epochs:int, global_step:int)->float:
    logger, writer = get_logger(), get_tb_writer()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    test_size = len(test_dl)

    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(test_dl):
            x, y = x.to(device), y.to(device, non_blocking=True)

            logits, _ = model(x)
            loss = lossfn(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            batch_size = x.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            if step % report_freq==0 or step==test_size-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}"
                    " Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, epochs, step, test_size-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, global_step)
    writer.add_scalar('val/top1', top1.avg, global_step)
    writer.add_scalar('val/top5', top5.avg, global_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(
        epoch+1, epochs, top1.avg))

    return top1.avg


def train_epoch(train_dl:DataLoader, model:nn.Module, device,
        lossfn:_Loss, optim:Optimizer, aux_weight:float, grad_clip:float,
        report_freq:int, epoch:int, epochs:int, global_step:int):
    logger, writer = get_logger(), get_tb_writer()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    train_size = len(train_dl)

    cur_lr = optim.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, global_step))
    writer.add_scalar('train/lr', cur_lr, global_step)

    model.train()
    for step, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device, non_blocking=True)
        batch_size = x.size(0)

        optim.zero_grad()
        if aux_weight > 0.:
            logits, aux_logits = model(x)
            loss = lossfn(logits, y)
            loss += aux_weight * lossfn(aux_logits, y)
        else:
            logits = model(x)
            loss = lossfn(logits, y)

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        if step % report_freq==0 or step==train_size-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, epochs, step, train_size-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/top1', prec1.item(), global_step)
        writer.add_scalar('train/top5', prec5.item(), global_step)
        global_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(
            epoch+1, epochs, top1.avg))

def train_test(train_dl:DataLoader, test_dl:DataLoader, model:nn.Module, device,
        train_lossfn:_Loss, test_lossfn:_Loss, optim:Optimizer, aux_weight:float, grad_clip:float,
        lr_scheduler:_LRScheduler, drop_path_prob:float, model_save_dir:str,
        report_freq:int, epochs:int):
    logger = get_logger()

    best_top1 = 0.
    for epoch in range(epochs):
        if drop_path_prob:
            drop_prob = drop_path_prob * epoch / epochs
            # set value as property in model (it will be used by forward())
            # this is necessory when using DataParallel(model)
            # https://github.com/pytorch/pytorch/issues/16885
            if hasattr(model, 'module'):
                model.module.drop_path_prob(drop_prob)
            else:
                model.drop_path_prob(drop_prob)
        global_step = epoch*len(train_dl)

        train_epoch(train_dl, model, device,
            train_lossfn, optim, aux_weight, grad_clip, report_freq, epoch,
            epochs, global_step)

        top1 = test_epoch(test_dl, model, device, test_lossfn,
            report_freq, epoch, epochs, global_step)

        lr_scheduler.step()

        # save
        if best_top1 < top1:
            best_top1, is_best = top1, True
        else:
            is_best = False
        utils.save_checkpoint(model, model_save_dir, is_best)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1