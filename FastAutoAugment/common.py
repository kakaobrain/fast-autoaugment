import logging
import warnings
import numpy as np
import os
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_logger(name='Fast AutoAugment', level=logging.DEBUG)->logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(os.path.expanduser(filepath))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# initializes random number gen, debugging etc
def common_init(logdir:str, dataroot:str, seed=42, detect_anomaly=True, log_level=logging.DEBUG) \
        -> Tuple[logging.Logger, str, str]:

    logger = get_logger('Fast AutoAugment')

    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(seed)

    if detect_anomaly:
        # TODO: enable below only in debug mode
        torch.autograd.set_detect_anomaly(True)

    logdir, dataroot = os.path.expanduser(logdir), os.path.expanduser(dataroot)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(dataroot, exist_ok=True)

    return logger, logdir, dataroot

def get_model_savepath(logdir, dataset, model, tag):
    return os.path.join(logdir, '%s_%s_%s.model' \
        % (dataset, model, tag))
