import logging
import warnings
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_logger(name, level=logging.DEBUG)->logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# initializes random number gen, debugging etc
def common_init(logdir:str, dataroot:str, seed=42, detect_anomaly=True, log_level=logging.DEBUG)->logging.Logger:
    # TODO: change name?
    logger = get_logger('Fast AutoAugment')

    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(seed)

    if detect_anomaly:
        # TODO: enable below only in debug mode
        torch.autograd.set_detect_anomaly(True)

    os.makedirs(os.path.expanduser(logdir), exist_ok=True)
    os.makedirs(os.path.expanduser(dataroot), exist_ok=True)

    return logger
