import logging
import numpy as np
import os
from typing import List

from ray.tune.trial_runner import TrialRunner # will be patched but not used
import gorilla


import torch
import torch.backends.cudnn as cudnn

from .config import Config
from .stopwatch import StopWatch
import yaml

_app_name = 'DefaultApp'

def _get_formatter():
    return logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

def get_logger(app_name=None)->logging.Logger:
    return logging.getLogger(app_name or _app_name)

def _setup_logger(app_name, level=logging.DEBUG)->logging.Logger:
    logger = logging.getLogger(app_name)
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
        app_name='DefaultApp', seed=42, detect_anomaly=True, log_level=logging.DEBUG) \
        -> Config:

    global _app_name
    _app_name = app_name

    conf = Config(config_filepath=config_filepath, defaults_filepath=defaults_filepath)

    assert not (conf['horovod'] and conf['only_eval']), 'can not use horovod when evaluation mode is enabled.'
    assert (conf['only_eval'] and conf['logdir']) or not conf['only_eval'], 'checkpoint path not provided in evaluation mode.'

    Config.set(conf)

    sw = StopWatch()
    StopWatch.set(sw)

    logger = _setup_logger(app_name)

    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(seed)

    if detect_anomaly:
        # TODO: enable below only in debug mode
        torch.autograd.set_detect_anomaly(True)

    logdir, dataroot = os.path.expanduser(conf['logdir']), os.path.expanduser(conf['dataroot'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(dataroot, exist_ok=True)
    conf['logdir'], conf['dataroot'] = logdir, dataroot

    # TODO: remove this in future
    if conf.get('decay', 0) > 0:
        logger.info('decay=%.4f' % conf['decay'])
        conf['optimizer']['decay'] = conf['decay']

    # file where logger would log messages
    logfile_path = os.path.join(logdir, '%s_%s_cv%.1f.log' % (conf['dataset'], conf['model']['type'],
            conf['cv_ratio']))
    _add_filehandler(logger, logfile_path)

    logger.info('configuration:')
    logger.info(conf.main_yaml)
    logger.info('--------------')
    logger.info('checkpoint will be saved at %s' % logdir)
    logger.info('Machine has {} gpus.'.format(torch.cuda.device_count()))

    return conf

def get_model_savepath(logdir, dataset, model, tag):
    return os.path.join(logdir, '%s_%s_%s.model' \
        % (dataset, model, tag))

