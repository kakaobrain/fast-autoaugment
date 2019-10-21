import logging
import warnings

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_logger(name, level=logging.DEBUG):
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
