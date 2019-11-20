import os
import json
from collections import defaultdict
import torch
import ray


from FastAutoAugment.common import get_logger, add_filehandler, common_init, get_model_savepath
from FastAutoAugment.aug_search import search_aug, train_no_aug
from theconf import Config as C, ConfigArgumentParser

# top1 metric from cross validation, this will contain dictionary of key, list.
top1_valid_by_cv = defaultdict(lambda: list)

if __name__ == '__main__':
    from pystopwatch2 import PyStopwatch
    sw = PyStopwatch()

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=200, help='number of samples to use for hyperopt search')
    parser.add_argument('--cv-ratio', type=float, default=0.4, help='split portion for test set, 0 to 1')
    parser.add_argument('--dataroot', type=str, default='~/torchvision_data_dir', help='torchvision data folder')
    parser.add_argument('--logdir', type=str, default='~/logdir')
    parser.add_argument('--cv-fold', type=int, default=0, help='Fold number to use (0 to 4)')
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str, default=None,
        help='reddis address of Ray cluster. Use None for single node run \
        otherwise it should something like host:6379. Make sure to run on head node: \
        "ray start --head --redis-port=6379"')
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true', help='if true, continue previous hyperopt search')
    parser.add_argument('--smoke-test', action='store_true', help='if specified then num_search is forced to be 4')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    logger = common_init(args.logdir, args.dataroot, args.seed)

    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay

    # file where logger would log messages
    logfile_path = os.path.join(args.logdir, '%s_%s_cv%.1f.log' % (C.get()['dataset'], C.get()['model']['type'],
            args.cv_ratio))
    add_filehandler(logger, logfile_path)

    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))

    logger.info('initialize ray...')
    ray.init(redis_address=args.redis,
        # allocate all GPUs on local node if cluster is not specified
        num_gpus=torch.cuda.device_count() if not args.redis else None)

    # after conducting N trials, we will chose the results of top num_result_per_cv
    num_result_per_cv = 10

    # cv_num must be same as n_splits=5 parameter in StratifiedShuffleSplit
    # TODO: remove this hard coding
    cv_num = 5

    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    train_no_aug(logger, sw, args.dataroot, args.logdir, cv_num, args.cv_ratio)

    search_aug(logger, sw, args.dataroot, args.logdir, args.num_policy, args.num_op, cv_num, args.cv_ratio,
        4 if args.smoke_test else args.num_search, num_result_per_cv,
        args.resume)

