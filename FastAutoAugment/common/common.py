import logging
import numpy as np
import os
from typing import List, Iterable, Union

from ray.tune.trial_runner import TrialRunner # will be patched but not used
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


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
    logdir = os.path.join(logdir, experiment_name)
    plotsdir = os.path.join(logdir, 'plots') if not conf['plotsdir'] \
        else os.path.expanduser(conf['plotsdir'])
    chkptdir = os.path.join(logdir, 'chkpt') if not conf['chkptdir'] \
        else os.path.expanduser(conf['chkptdir'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(plotsdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)

    conf['logdir'], conf['dataroot'] = logdir, dataroot
    conf['plotsdir'], conf['chkptdir'] = plotsdir, chkptdir

    # copy net config to experiment folder for reference
    with open(os.path.join(logdir, 'full_config.yaml'), 'w') as f:
        yaml.dump(conf, f, default_flow_style=False)

    # file where logger would log messages
    logfilename = 'logs.log'
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
