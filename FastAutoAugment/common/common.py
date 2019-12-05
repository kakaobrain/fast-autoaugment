import logging
import numpy as np
import os
from typing import List, Iterable, Union

from ray.tune.trial_runner import TrialRunner # will be patched but not used
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from  torch.utils.data import DataLoader
import  torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from .config import Config
from .stopwatch import StopWatch
from .metrics import SummaryWriterDummy
from . import utils

SummaryWriterAny = Union[SummaryWriterDummy, SummaryWriter]

_app_name = 'DefaultApp'
_tb_writer:SummaryWriterAny = None

def _get_formatter()->logging.Formatter:
    return logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

def get_logger(experiment_name=None)->logging.Logger:
    return logging.getLogger(experiment_name or _app_name)

def get_tb_writer()->SummaryWriterAny:
    global _tb_writer
    return _tb_writer

def _setup_logger(experiment_name, level=logging.DEBUG)->logging.Logger:
    logger = logging.getLogger(experiment_name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    ch.setFormatter(_get_formatter())
    logger.addHandler(ch)
    return logger

def _add_filehandler(logger, filepath):
    fh = logging.FileHandler(filename=os.path.expanduser(filepath))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_get_formatter())
    logger.addHandler(fh)

# initializes random number gen, debugging etc
def common_init(config_filepath:str, defaults_filepath:str,
        param_args:List[str]=[], experiment_name='', seed=42, detect_anomaly=True,
        log_level=logging.DEBUG, is_master=True, tb_names:Iterable[str]=['0']) \
        -> Config:

    global _app_name
    _app_name = experiment_name

    conf = Config(config_filepath=config_filepath, defaults_filepath=defaults_filepath)

    assert not (conf['horovod'] and conf['only_eval']), 'can not use horovod when evaluation mode is enabled.'
    assert (conf['only_eval'] and conf['logdir']) or not conf['only_eval'], 'checkpoint path not provided in evaluation mode.'

    Config.set(conf)

    sw = StopWatch()
    StopWatch.set(sw)

    logger = _setup_logger(experiment_name)

    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if detect_anomaly:
        # TODO: enable below only in debug mode
        torch.autograd.set_detect_anomaly(True)

    logdir = os.path.expanduser(conf['logdir'])
    dataroot = os.path.expanduser(conf['dataroot'])
    plotsdir = os.path.expanduser(conf['plotsdir'])
    if experiment_name:
        logdir = os.path.join(logdir, experiment_name)
        plotsdir = os.path.join(plotsdir, experiment_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(plotsdir, exist_ok=True)
    conf['logdir'], conf['dataroot'] = logdir, dataroot
    conf['plotsdir'] = plotsdir

    # copy net config to experiment folder for reference
    with open(os.path.join(logdir, 'full_config.yaml'), 'w') as f:
        yaml.dump(conf, f, default_flow_style=False)

    # file where logger would log messages
    logfilename = '{}_cv{:.1f}.log'.format(conf['dataset'],
            conf['val_ratio'])
    logfile_path = os.path.join(logdir, logfilename)
    _add_filehandler(logger, logfile_path)

    logger.info('checkpoint will be saved at %s' % logdir)
    logger.info('Machine has {} gpus.'.format(torch.cuda.device_count()))
    logger.info('Original CUDA_VISIBLE_DEVICES: {}'.format( \
            os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet'))

    if conf['gpus'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(conf['gpus'])
        logger.info('Only these GPUs will be used: {}'.format(conf['gpus']))
        # alternative: torch.cuda.set_device(config.gpus[0])

    gpu_usage = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')
    for i, line in enumerate(gpu_usage):
        vals = line.split(',')
        if len(vals) == 2:
            logger.info('GPU {} mem: {}, used: {}'.format(i, vals[0], vals[1]))

    global _tb_writer
    _tb_writer = _create_tb_writer(conf, is_master, tb_names)

    return conf

def get_model_savepath(logdir, dataset, model, tag):
    return os.path.join(logdir, '%s_%s_%s.model' \
        % (dataset, model, tag))

def _create_tb_writer(conf:Config, is_master=True,
        tb_names:Iterable[str]=['0'])->SummaryWriterAny:
    WriterClass = SummaryWriterDummy if not conf['enable_tb'] or not is_master \
            else SummaryWriter

    return WriterClass(log_dir='{}/tb/{}'.format(conf['logdir']))

def test_epoch(test_dl:DataLoader, model:nn.Module, device, criterion:_Loss,
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
            loss = criterion(logits, y)

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
        criterion:_Loss, optim:Optimizer, aux_weight:float, grad_clip:float,
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
            loss = criterion(logits, y)
            loss += aux_weight * criterion(aux_logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

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
        criterion:_Loss, optim:Optimizer, aux_weight:float, grad_clip:float,
        lr_scheduler:_LRScheduler, drop_path_prob:float, model_save_dir:str,
        report_freq:int, epoch:int, epochs:int):
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
            criterion, optim, aux_weight, grad_clip, report_freq, epoch,
            epochs, global_step)

        top1 = test_epoch(test_dl, model, device, criterion,
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