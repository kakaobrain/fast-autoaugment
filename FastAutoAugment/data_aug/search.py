from hyperopt import hp
from ray.tune.trial_runner import TrialRunner # will be patched but not used
import gorilla
from ray.tune.trial import Trial
from ray.tune.suggest import HyperOptSearch
from ray.tune import register_trainable, run_experiments
import numpy as np
import copy
import json
from collections import OrderedDict
import ray
import torch
import time
import os
from tqdm import tqdm

from ..common.aug_policies import remove_deplicates, policy_decoder
from ..common.data import get_dataloaders
from ..common.metrics import Accumulator
from ..networks import get_model, num_class
from ..common.augmentations import augment_list
from .train import train_and_eval
from ..common.common import get_model_savepath, get_logger

from ..common.config import Config
from ..common.stopwatch import StopWatch


# this method is overriden version of ray.tune.trial_runner.TrialRunner.step using monkey patching
def _step_w_log(self):
    logger = get_logger()

    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner,
        'step')

    # collect counts by status for all trials
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED,
            Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt

    # get the best top1 accuracy from all finished trials so far
    best_top1_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result: # TODO: why would this happen?
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])

    # display best accuracy from all finished trial
    logger.info('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')

    # call original step method
    return original(self)

# override ray.tune.trial_runner.TrialRunner.step method so we can print best accuracy at each step
patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', _step_w_log,
    settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)



@ray.remote(num_gpus=torch.cuda.device_count(), max_calls=1)
def _train_model(conf, dataroot, augment, val_ratio, val_fold, save_path=None,
        only_eval=False):
    Config.set(conf)
    conf['autoaug']['loader']['aug'] = augment
    model_type = conf['autoaug']['model']['type']

    result = train_and_eval(conf, val_ratio=val_ratio, val_fold=val_fold,
        save_path=save_path, only_eval=only_eval)
    return model_type, val_fold, result

def _train_no_aug(conf):
    logger, sw = get_logger(), StopWatch.get()

    # region conf vars
    conf_data     = conf['dataset']
    dataroot    = conf['dataroot']
    logdir      = conf['logdir']
    redis_ip    = conf['redis']
    conf_loader = conf['autoaug']['loader']
    conf_model  = conf['autoaug']['model']
    model_type  = conf_model['type']
    ds_name     = conf_data['name']
    aug         = conf_loader['aug']
    cutout      = conf_loader['cutout']
    val_ratio   = conf_loader['val_ratio']
    batch_size  = conf_loader['batch']
    epochs      = conf_loader['epochs']
    val_fold    = conf_loader['val_fold']
    cv_num      = conf_loader['cv_num']
    # endregion

    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----'\
         % (cv_num, val_ratio))
    sw.start(tag='train_no_aug')

    # for each fold, we will save model
    save_paths = [get_model_savepath(logdir, ds_name, model_type,
        'ratio%.1f_fold%d' % (val_ratio, i)) for i in range(cv_num)]

    # Train model for each fold, save model in specified path, put result
    # in reqs list. These models are trained with aug specified in config.
    # TODO: configuration will be changed ('aug' key),
    #   but do we really need deepcopy everywhere?
    reqs = [
        # TODO: eliminate need for deep copy as only aug key is changed
        _train_model.remote(copy.deepcopy(copy.deepcopy(conf)), dataroot,
            aug, val_ratio, i,save_path=save_paths[i], only_eval=True)
        for i in range(cv_num)]

    # we now probe saved models for each fold to check the epoch number
    # they are on. When every fold crosses an epoch number, we update
    # the progress.
    tqdm_epoch = tqdm(range(epochs))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs_per_cv = OrderedDict()
            for cv_idx in range(cv_num):
                try:
                    if os.path.exists(save_paths[cv_idx]):
                        latest_ckpt = torch.load(save_paths[cv_idx])
                        if 'epoch' not in latest_ckpt:
                            epochs_per_cv['cv%d' % (cv_idx + 1)] = epochs
                            continue
                    else:
                        continue
                    epochs_per_cv['cv%d' % (cv_idx+1)] = latest_ckpt['epoch']
                except Exception as e:
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= epochs:
                is_done = True
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    pretrain_results = ray.get(reqs)
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' %
            (r_model, r_cv+1, r_dict['top1_train'], r_dict['top1_valid']))
    logger.info('processed in %.4f secs' % sw.pause('train_no_aug'))

def search(conf):
    logger, sw = get_logger(), StopWatch.get()

    # region conf vars
    conf_data     = conf['dataset']
    dataroot    = conf['dataroot']
    logdir      = conf['logdir']
    redis_ip    = conf['redis']
    conf_loader = conf['autoaug']['loader']
    conf_model  = conf['autoaug']['model']
    model_type  = conf_model['type']
    ds_name     = conf_data['name']
    aug         = conf_loader['aug']
    val_ratio   = conf_loader['val_ratio']
    epochs      = conf_loader['epochs']
    val_fold    = conf_loader['val_fold']
    cv_num      = conf_loader['cv_num']
    num_policy = conf['autoaug']['num_policy']
    num_op = conf['autoaug']['num_op']
    num_search = conf['autoaug']['num_search']
    num_result_per_cv = conf['autoaug']['num_result_per_cv']
    smoke_test = conf['smoke_test']
    resume = conf['resume']
    # endregion

    ray.init(redis_address=redis_ip,
        # allocate all GPUs on local node if cluster is not specified
        num_gpus=torch.cuda.device_count() if not redis_ip else None)

    # first train with no aug
    _train_no_aug(conf)

    # get values from config
    num_samples = 4 if smoke_test else num_search

    logger.info('----- Search Test-Time Augmentation Policies -----')
    sw.start(tag='search')

    save_paths = [get_model_savepath(logdir, ds_name,
        model_type, 'ratio%.1f_fold%d' %
            (val_ratio, i)) for i in range(cv_num)]

    copied_c = copy.deepcopy(conf)
    ops = augment_list(False)
    space = {}
    for i in range(num_policy):
        for j in range(num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' %
                (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' %
                (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' %
                (i, j), 0.0, 1.0)

    final_policy_set = []
    total_computation = 0
    reward_attr = 'top1_valid'      # top1_valid or minus_loss
    for _ in range(1):  # run multiple times.
        for val_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (ds_name,
                model_type, val_fold, val_ratio)
            #logger.info(name)
            register_trainable(name, (lambda augs,
                rpt: _eval_tta(copy.deepcopy(copied_c), augs, rpt)))
            algo = HyperOptSearch(space, max_concurrent=4*20,
                reward_attr=reward_attr)

            exp_config = {
                name: {
                    'run': name,
                    'num_samples': num_samples,
                    'resources_per_trial': {'gpu': 1},
                    'stop': {'training_iteration': num_policy},
                    'config': {
                        'dataroot': dataroot, 'save_path': save_paths[val_fold],
                        'val_ratio': val_ratio, 'val_fold': val_fold,
                        'num_op': num_op, 'num_policy': num_policy
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo,
                scheduler=None, verbose=0, queue_trials=True,
                resume=resume, raise_on_failed_trial=False)

            results = [x for x in results if x.last_result is not None]
            results = sorted(results, key=lambda x: x.last_result[reward_attr],
                reverse=True)

            # calculate computation usage
            for result in results:
                total_computation += result.last_result['elapsed_time']

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(result.config, num_policy, num_op)
                logger.info('loss=%.12f top1_valid=%.4f %s' %
                    (result.last_result['minus_loss'],
                        result.last_result['top1_valid'], final_policy))

                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)

    logger.info(json.dumps(final_policy_set))
    logger.info('final_policy=%d' % len(final_policy_set))
    logger.info('processed in %.4f secs, gpu hours=%.4f' % (sw.pause('search'), total_computation / 3600.))
    logger.info('----- Train with Augmentations model=%s dataset=%s aug=%s ratio(test)=%.1f -----' \
        % (model_type, ds_name, aug, val_ratio))
    sw.start(tag='train_aug')

    num_experiments = 5
    default_path = [get_model_savepath(logdir, ds_name, model_type, 'ratio%.1f_default%d'  \
        % (val_ratio, _)) for _ in range(num_experiments)]
    augment_path = [get_model_savepath(logdir, ds_name, model_type, 'ratio%.1f_augment%d'  \
        % (val_ratio, _)) for _ in range(num_experiments)]
    reqs = [_train_model.remote(copy.deepcopy(copied_c), dataroot, aug, 0.0, 0, save_path=default_path[_], only_eval=True) \
        for _ in range(num_experiments)] + \
        [_train_model.remote(copy.deepcopy(copied_c), dataroot, final_policy_set, 0.0, 0, save_path=augment_path[_]) \
            for _ in range(num_experiments)]

    tqdm_epoch = tqdm(range(epochs))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(num_experiments):
                try:
                    if os.path.exists(default_path[exp_idx]):
                        latest_ckpt = torch.load(default_path[exp_idx])
                        epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass
                try:
                    if os.path.exists(augment_path[exp_idx]):
                        latest_ckpt = torch.load(augment_path[exp_idx])
                        epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass

            tqdm_epoch.set_postfix(epochs)
            if len(epochs) == num_experiments*2 and min(epochs.values()) >= epochs:
                is_done = True
            if len(epochs) == num_experiments*2 and min(epochs.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    final_results = ray.get(reqs)

    for train_mode in ['default', 'augment']:
        avg = 0.
        for _ in range(num_experiments):
            r_model, r_cv, r_dict = final_results.pop(0)
            logger.info('[%s] top1_train=%.4f top1_test=%.4f' % (train_mode, r_dict['top1_train'], r_dict['top1_test']))
            avg += r_dict['top1_test']
        avg /= num_experiments
        logger.info('[%s] top1_test average=%.4f (#experiments=%d)' % (train_mode, avg, num_experiments))
    logger.info('processed in %.4f secs' % sw.pause('train_aug'))

    logger.info(sw)


def _eval_tta(conf, augment, reporter):
    Config.set(conf)

    # region conf vars
    conf_data     = conf['dataset']
    conf_loader = conf['autoaug']['loader']
    conf_model  = conf['autoaug']['model']
    ds_name     = conf_data['name']
    cutout   = conf_loader['cutout']
    batch_size   = conf_loader['batch_size']
    n_workers     = conf_loader['n_workers']
    # endregion

    val_ratio, val_fold, save_path = \
        augment['val_ratio'], augment['val_fold'], augment['save_path']

    # setup - provided augmentation rules
    aug = policy_decoder(augment, augment['num_policy'], augment['num_op'])

    # eval
    model = get_model(conf_model, num_class(ds_name))
    ckpt = torch.load(save_path)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    loaders = []
    for _ in range(augment['num_policy']):
        tl, validloader, tl2, _ = get_dataloaders(ds_name, batch_size,
            augment['dataroot'], aug, cutout,
            load_train=True, load_test=True,
            val_ratio=val_ratio, val_fold=val_fold, n_workers=n_workers)
        loaders.append(iter(validloader))
        del tl, tl2 # TODO: why exclude validloader?

    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)
                data, label = data.cuda(), label.cuda()

                pred = model(data)

                loss = loss_fn(pred, label)
                losses.append(loss.detach().cpu().numpy())

                _, pred = pred.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(label.view(1, -1).expand_as(pred)) \
                    .detach().cpu().numpy()
                corrects.append(correct)
                del loss, correct, pred, data, label

            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()

            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict({
                'minus_loss': -1 * np.sum(losses_min),
                'correct': np.sum(corrects_max),
                'cnt': len(corrects_max)
            })
            del corrects, corrects_max
    except StopIteration:
        pass

    del model
    metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'],
        elapsed_time=gpu_secs, done=True)
    return metrics['correct']

