import logging
import numpy as np
import os
from typing import List, Iterable, Union, Optional

import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .stopwatch import StopWatch


class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


SummaryWriterAny = Union[SummaryWriterDummy, SummaryWriter]

_app_name = 'DefaultApp'
_tb_writer: SummaryWriterAny = None


def _get_formatter() -> logging.Formatter:
    return logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


def get_logger(experiment_name=None) -> logging.Logger:
    return logging.getLogger(experiment_name or _app_name)


def get_tb_writer() -> SummaryWriterAny:
    global _tb_writer
    return _tb_writer


def _setup_logger(experiment_name, level=logging.DEBUG) -> logging.Logger:
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


def _setup_dirs(conf_common: Config, conf_data: Config, experiment_name: str):
    logger = get_logger()

    dataroot = os.path.expanduser(conf_data['dataroot'])
    os.makedirs(dataroot, exist_ok=True)

    logdir = conf_common['logdir']
    if logdir:
        logdir = os.path.expanduser(logdir)
        logdir = os.path.join(logdir, experiment_name)
        plotsdir = os.path.join(logdir, 'plots') if not conf_common['plotsdir'] \
            else os.path.expanduser(conf_common['plotsdir'])
        chkptdir = os.path.join(logdir, 'chkpt') if not conf_common['chkptdir'] \
            else os.path.expanduser(conf_common['chkptdir'])
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(plotsdir, exist_ok=True)
        os.makedirs(chkptdir, exist_ok=True)

        # file where logger would log messages
        logfilename = 'logs.log'
        logfile_path = os.path.join(logdir, logfilename)
        _add_filehandler(logger, logfile_path)
        logger.info('logdir: %s' % logdir)
    else:
        logger.warn('logdir not specified, no logs will be created or any models saved')
        plotsdir = chkptdir = None

    conf_common['logdir'], conf_data['dataroot'] = logdir, dataroot
    conf_common['plotsdir'], conf_common['chkptdir'] = plotsdir, chkptdir

def _setup_gpus(conf_common):
    logger = get_logger()
    if conf_common['gpus'] is not None:
        csv = str(conf_common['gpus'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(conf_common['gpus'])
        torch.cuda.set_device(int(csv.split(',')[0]))
        logger.info('Only these GPUs will be used: {}'.format(
            conf_common['gpus']))
        # alternative: torch.cuda.set_device(config.gpus[0])

    seed = conf_common['seed']

    cudnn.enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

    if conf_common['detect_anomaly']:
        logger.warn(
            'PyTorch code will be 6X slower because detect_anomaly=True.')
        torch.autograd.set_detect_anomaly(True)

    logger.info('Machine has {} gpus.'.format(torch.cuda.device_count()))
    logger.info('Original CUDA_VISIBLE_DEVICES: {}'.format(
        os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet'))

    gpu_usage = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')
    for i, line in enumerate(gpu_usage):
        vals = line.split(',')
        if len(vals) == 2:
            logger.info('GPU {} mem: {}, used: {}'.format(i, vals[0], vals[1]))


# initializes random number gen, debugging etc
def common_init(config_filepath: Optional[str], defaults_filepath: Optional[str],
                param_args: List[Optional[str]] = [], experiment_name='',
                log_level=logging.DEBUG, is_master=True) \
        -> Config:

    global _app_name
    _app_name = experiment_name

    conf = Config(config_filepath=config_filepath,
                  defaults_filepath=defaults_filepath,
                  use_args=True)

    Config.set(conf)

    sw = StopWatch()
    StopWatch.set(sw)

    logger = _setup_logger(experiment_name)
    conf_common = conf['common']
    conf_data = conf['dataset']
    _setup_dirs(conf_common, conf_data, experiment_name)
    _setup_gpus(conf_common)

    logdir = conf_common['logdir']
    if logdir:
        # copy net config to experiment folder for reference
        with open(os.path.join(logdir, 'full_config.yaml'), 'w') as f:
            yaml.dump(conf, f, default_flow_style=False)

    global _tb_writer
    _tb_writer = _create_tb_writer(conf_common, is_master)

    return conf


def get_model_savepath(logdir, dataset, model, tag):
    return os.path.join(logdir, '%s_%s_%s.model'
                        % (dataset, model, tag))


def _create_tb_writer(conf_common: Config, is_master=True)\
        -> SummaryWriterAny:
    logdir = conf_common['logdir']
    WriterClass = SummaryWriterDummy if not conf_common['enable_tb'] or \
                                        not is_master or \
                                        not logdir \
        else SummaryWriter

    return WriterClass(log_dir=os.path.join(logdir, 'tb'))
