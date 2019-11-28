import logging
import numpy as np
import os
from typing import List, Iterable

from ray.tune.trial_runner import TrialRunner # will be patched but not used
import yaml

import torch
import torch.backends.cudnn as cudnn

from .config import Config
from .stopwatch import StopWatch


_app_name = 'DefaultApp'

def _get_formatter():
    return logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

def get_logger(experiment_name=None)->logging.Logger:
    return logging.getLogger(experiment_name or _app_name)

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
def common_init(config_filepath:str, defaults_filepath:str, param_args:List[str]=[],
        experiment_name='', seed=42, detect_anomaly=True, log_level=logging.DEBUG) \
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

    logdir, dataroot = os.path.expanduser(conf['logdir']), os.path.expanduser(conf['dataroot'])
    if experiment_name:
        logdir = os.path.join(logdir, experiment_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(dataroot, exist_ok=True)
    conf['logdir'], conf['dataroot'] = logdir, dataroot

    # copy net config to experiment folder for reference
    with open(os.path.join(logdir, 'full_config.yaml'), 'w') as f:
        yaml.dump(conf, f, default_flow_style=False)

    # file where logger would log messages
    logfilename = '{}_cv{:.1f}.log'.format(conf['dataset'],
            conf['cv_ratio'])
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

    return conf

def get_model_savepath(logdir, dataset, model, tag):
    return os.path.join(logdir, '%s_%s_%s.model' \
        % (dataset, model, tag))

def create_tb_writers(conf:Config, is_master=True, sub_folders:Iterable[str]=['0']):
    if not conf['enable_tb'] or not is_master:
        # create dummy writer that will ignore all writes
        from .metrics import SummaryWriterDummy as SummaryWriter
    else:
        from torch.utils.tensorboard import SummaryWriter
    return [SummaryWriter(log_dir=f'{log_dir}/tb/{x}') for x in sub_folders]



