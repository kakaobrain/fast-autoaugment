import itertools
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

from tqdm import tqdm

from .common import get_logger
from .data import get_dataloaders
from .lr_scheduler import adjust_learning_rate_pyramid, adjust_learning_rate_resnet
from .metrics import accuracy, Accumulator
from .networks import get_model, num_class

from warmup_scheduler import GradualWarmupScheduler


# TODO: remove scheduler parameter?
def run_epoch(conf, logger, model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1,
        scheduler=None):
    """Runs epoch for given dataloader and model. If optimizer is supplied then backprop and model
    update is done as well. This can be called from test to train modes.
    """

    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))  #TODO: remove?
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, conf['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        preds = model(data)

        # Try to visualize model via tensorwatch
        # tw.draw_model(model, tuple(data.shape)).save('/home/dedey/model.png')

        # Try to visualize model via tensorboard
        # writer.add_graph(model, data)
        # logger.info('Just wrote model file!')
        loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            if getattr(optimizer, "synchronize", None):
                optimizer.synchronize()     # for horovod
            # grad clipping defaults to 5 (same as Darts)
            if conf['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), conf['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        # below changes LR for every batch in epoch
        # TODO: should we do LR step at epoch start only?
        # if scheduler is not None:
        #     scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, conf['epoch'],
                metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, conf['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics

# metric could be 'last', 'test', 'val', 'train'.
def train_and_eval(conf, cv_ratio, cv_fold, save_path, only_eval, tb_tag=None, reporter=None, metric='test'):

    logger = get_logger()

    log_dir = conf['logdir']
    dataroot = conf['dataroot']
    horovod = conf['horovod']
    checkpoint_freq = conf['checkpoint_freq']

    # initialize horovod
    # TODO: move to common init
    if horovod:
        import horovod.torch as hvd
        hvd.init()
        device = torch.device('cuda', hvd.local_rank())
        torch.cuda.set_device(device)

    if not reporter:
        reporter = lambda **kwargs: 0

    # get dataloaders with transformations and splits applied
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(conf['dataset'], conf['batch'], dataroot,
        conf['aug'], cv_ratio=cv_ratio, cv_fold=cv_fold, horovod=horovod)

    # create a model & an optimizer
    model = get_model(conf['model'], num_class(conf['dataset']), data_parallel=(not horovod))

    # select loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if conf['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=conf['lr'],
            momentum=conf['optimizer'].get('momentum', 0.9),
            weight_decay=conf['optimizer']['decay'],
            nesterov=conf['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % conf['optimizer']['type'])

    # distributed optimizer if horovod is used
    is_master = True
    if horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        optimizer._requires_update = set()  # issue : https://github.com/horovod/horovod/issues/1099
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        if hvd.rank() != 0:
            is_master = False
    logger.debug('is_master=%s' % is_master)

    # select LR schedule
    lr_scheduler_type = conf['lr_schedule'].get('type', 'cosine')
    scheduler = None
    if lr_scheduler_type == 'cosine':
        t_max = conf['epoch']
        # adjust max epochs for warmup
        # TODO: shouldn't we be increasing t_max or schedule lr only after warmup?
        if conf['lr_schedule'].get('warmup', None):
            t_max -= conf['lr_schedule']['warmup']['epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'pyramid':
        scheduler = adjust_learning_rate_pyramid(optimizer, conf['epoch'])
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)
    # select warmup for LR schedule
    if conf['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=conf['lr_schedule']['warmup']['multiplier'],
            total_epoch=conf['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    # create tensorboard writers
    if not tb_tag or not is_master:
        # create dummy writer that will ignore all writes
        from FastAutoAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tb_tag not provided, no tensorboard log.')
    else:
        from torch.utils.tensorboard import SummaryWriter
    writers = [SummaryWriter(log_dir=f'{log_dir}/logs/{tb_tag}/{x}') for x in ['train', 'valid', 'test']]

    result = OrderedDict()
    epoch_start = 1
    # if model available from previous checkpount then load it
    if save_path and os.path.exists(save_path):
        logger.info('%s checkpoint found. loading...' % save_path)
        data = torch.load(save_path)

        # wehn checkpointing we do add 'model' key so other cases are special cases
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])

            # TODO: do we need change here?
            if not isinstance(model, DataParallel):
                # for non-dataparallel models, remove default 'module.' prefix
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                # for dataparallel models, make sure 'module.' prefix exist
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})

            # load optimizer
            optimizer.load_state_dict(data['optimizer'])

            # restore epoch count
            if data['epoch'] < conf['epoch']:
                epoch_start = data['epoch']
            else:
                # epochs finished, switch to eval mode
                only_eval = False

        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('model checkpoint does not exist at "%s". skip to pretrain weights...' % save_path)
        only_eval = False # we made attempt to load checkpt but as it does not exist, switch to train mode

    # if eval only then run model on train, test and val sets
    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict() # stores metrics for each set
        rs['train'] = run_epoch(conf, logger, model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
        rs['valid'] = run_epoch(conf, logger, model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
        rs['test'] = run_epoch(conf, logger, model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1, best_valid_loss = 0, 10.0e10
    max_epoch = conf['epoch']
    for epoch in range(epoch_start, max_epoch + 1):
        if horovod:
            trainsampler.set_epoch(epoch)

        # run train epoch and update the model
        model.train()
        rs = dict()
        rs['train'] = run_epoch(conf, logger, model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=is_master, scheduler=scheduler)
        if scheduler:
            scheduler.step()

        model.eval()

        # check for nan loss
        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        # collect metrics on val and test set, checkpoint
        if epoch % checkpoint_freq == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(conf, logger, model, validloader, criterion, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=is_master)
            rs['test'] = run_epoch(conf, logger, model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=is_master)

            # TODO: is this good enough condition?
            if rs[metric]['loss'] < best_valid_loss or rs[metric]['top1'] > best_top1:
                best_top1 = rs[metric]['top1']
                best_valid_loss = rs[metric]['loss']
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
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)

    del model

    result['top1_test'] = best_top1
    return result

