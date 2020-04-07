import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import itertools
import json
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from FastAutoAugment.common import get_logger, EMA, add_filehandler
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.lr_scheduler import adjust_learning_rate_resnet
from FastAutoAugment.metrics import accuracy, Accumulator, CrossEntropyLabelSmooth
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.tf_port.rmsprop import RMSpropTF
from FastAutoAugment.aug_mixup import CrossEntropyMixUpLabelSmooth, mixup
from warmup_scheduler import GradualWarmupScheduler

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)


def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None, is_master=True, ema=None, wd=0.0, tqdm_disabled=False):
    if verbose:
        loader = tqdm(loader, disable=tqdm_disabled)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    params_without_bn = [params for name, params in model.named_parameters() if not ('_bn' in name or '.bn' in name)]

    loss_ema = None
    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if C.get().conf.get('mixup', 0.0) <= 0.0 or optimizer is None:
            preds = model(data)
            loss = loss_fn(preds, label)
        else:   # mixup
            data, targets, shuffled_targets, lam = mixup(data, label, C.get()['mixup'])
            preds = model(data)
            loss = loss_fn(preds, targets, shuffled_targets, lam)
            del shuffled_targets, lam

        if optimizer:
            loss += wd * (1. / 2.) * sum([torch.sum(p ** 2) for p in params_without_bn])
            loss.backward()
            grad_clip = C.get()['optimizer'].get('clip', 5.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema(model, (epoch - 1) * total_steps + steps)

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if loss_ema:
            loss_ema = loss_ema * 0.9 + loss.item() * 0.1
        else:
            loss_ema = loss.item()
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            postfix['loss_ema'] = loss_ema
            loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disabled and verbose:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, local_rank=-1, evaluation_interval=5):
    total_batch = C.get()["batch"]
    if local_rank >= 0:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=int(os.environ['WORLD_SIZE']))
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)

        C.get()['lr'] *= dist.get_world_size()
        logger.info(f'local batch={C.get()["batch"]} world_size={dist.get_world_size()} ----> total batch={C.get()["batch"] * dist.get_world_size()}')
        total_batch = C.get()["batch"] * dist.get_world_size()

    is_master = local_rank < 0 or dist.get_rank() == 0
    if is_master:
        add_filehandler(logger, args.save + '.log')

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, multinode=(local_rank >= 0))

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']), local_rank=local_rank)
    model_ema = get_model(C.get()['model'], num_class(C.get()['dataset']), local_rank=-1)
    model_ema.eval()

    criterion_ce = criterion = CrossEntropyLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))
    if C.get().conf.get('mixup', 0.0) > 0.0:
        criterion = CrossEntropyMixUpLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=0.0,
            nesterov=C.get()['optimizer'].get('nesterov', True)
        )
    elif C.get()['optimizer']['type'] == 'rmsprop':
        optimizer = RMSpropTF(
            model.parameters(),
            lr=C.get()['lr'],
            weight_decay=0.0,
            alpha=0.9, momentum=0.9,
            eps=0.001
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'efficientnet':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.97 ** int((x + C.get()['lr_schedule']['warmup']['epoch']) / 2.4))
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None) and C.get()['lr_schedule']['warmup']['epoch'] > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if not tag or not is_master:
        from FastAutoAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    if C.get()['optimizer']['ema'] > 0.0 and is_master:
        # https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4?u=ildoonet
        ema = EMA(C.get()['optimizer']['ema'])
    else:
        ema = None

    result = OrderedDict()
    epoch_start = 1
    if save_path != 'test.pth':     # and is_master: --> should load all data(not able to be broadcasted)
        if save_path and os.path.exists(save_path):
            logger.info('%s file found. loading...' % save_path)
            data = torch.load(save_path)
            key = 'model' if 'model' in data else 'state_dict'

            if 'epoch' not in data:
                model.load_state_dict(data)
            else:
                logger.info('checkpoint epoch@%d' % data['epoch'])
                if not isinstance(model, (DataParallel, DistributedDataParallel)):
                    model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
                else:
                    model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
                logger.info('optimizer.load_state_dict+')
                optimizer.load_state_dict(data['optimizer'])
                if data['epoch'] < C.get()['epoch']:
                    epoch_start = data['epoch']
                else:
                    only_eval = True
                if ema is not None:
                    ema.shadow = data.get('ema', {}) if isinstance(data.get('ema', {}), dict) else data['ema'].state_dict()
            del data
        else:
            logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
            if only_eval:
                logger.warning('model checkpoint not found. only-evaluation mode is off.')
            only_eval = False

    if local_rank >= 0:
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        logger.info(f'multinode init. local_rank={dist.get_rank()} is_master={is_master}')
        torch.cuda.synchronize()

    tqdm_disabled = bool(os.environ.get('TASK_NAME', '')) and local_rank != 0  # KakaoBrain Environment

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0], is_master=is_master)

        with torch.no_grad():
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1], is_master=is_master)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2], is_master=is_master)
            if ema is not None and len(ema) > 0:
                model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                rs['valid'] = run_epoch(model_ema, validloader, criterion_ce, None, desc_default='valid(EMA)', epoch=0, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                rs['test'] = run_epoch(model_ema, testloader_, criterion_ce, None, desc_default='*test(EMA)', epoch=0, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        if local_rank >= 0:
            trainsampler.set_epoch(epoch)

        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=(is_master and local_rank <= 0), scheduler=scheduler, ema=ema, wd=C.get()['optimizer']['decay'], tqdm_disabled=tqdm_disabled)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if ema is not None and C.get()['optimizer']['ema_interval'] > 0 and epoch % C.get()['optimizer']['ema_interval'] == 0:
            logger.info(f'ema synced+ rank={dist.get_rank()}')
            if ema is not None:
                model.load_state_dict(ema.state_dict())
            for name, x in model.state_dict().items():
                # print(name)
                dist.broadcast(x, 0)
            torch.cuda.synchronize()
            logger.info(f'ema synced- rank={dist.get_rank()}')

        if is_master and (epoch % evaluation_interval == 0 or epoch == max_epoch):
            with torch.no_grad():
                rs['valid'] = run_epoch(model, validloader, criterion_ce, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                rs['test'] = run_epoch(model, testloader_, criterion_ce, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)

                if ema is not None:
                    model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                    rs['valid'] = run_epoch(model_ema, validloader, criterion_ce, None, desc_default='valid(EMA)', epoch=epoch, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                    rs['test'] = run_epoch(model_ema, testloader_, criterion_ce, None, desc_default='*test(EMA)', epoch=epoch, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)

            logger.info(
                f'epoch={epoch} '
                f'[train] loss={rs["train"]["loss"]:.4f} top1={rs["train"]["top1"]:.4f} '
                f'[valid] loss={rs["valid"]["loss"]:.4f} top1={rs["valid"]["top1"]:.4f} '
                f'[test] loss={rs["test"]["loss"]:.4f} top1={rs["test"]["top1"]:.4f} '
            )

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if is_master and save_path:
                    logger.info('save model@%d to %s, err=%.4f' % (epoch, save_path, 1 - best_top1))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'ema': ema.state_dict() if ema is not None else None,
                    }, save_path)

    del model

    result['top1_test'] = best_top1
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='test.pth')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation-interval', type=int, default=5)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, local_rank=args.local_rank, metric='test', evaluation_interval=args.evaluation_interval)
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
