import torch
import yaml

from ..common import utils

from ..common.trainer import Trainer
from ..common.config import Config
from ..common.common import get_logger, logdir_abspath
from .model_desc import RunMode
from . import nas_utils

def eval_arch(conf_eval:Config):
    logger = get_logger()

    # region conf vars
    conf_loader       = conf_eval['loader']
    model_desc_file = conf_eval['model_desc_file']
    save_filename    = conf_eval['save_filename']
    conf_model_desc   = conf_eval['model_desc']
    conf_train = conf_eval['trainer']
    # endregion

    # load model desc file to get template model
    model_desc_filepath = logdir_abspath(model_desc_file)
    assert model_desc_filepath
    with open(model_desc_filepath, 'r') as f:
        template_model_desc = yaml.load(f, Loader=yaml.Loader)

    device = torch.device(conf_eval['device'])

    # create model
    model = nas_utils.create_model(conf_model_desc, device,
                                   run_mode=RunMode.EvalTrain,
                                   template_model_desc=template_model_desc)

    # get data
    train_dl, test_dl = nas_utils.get_train_test_data(conf_loader)

    trainer = Trainer(conf_train, model, device)
    trainer.fit(train_dl, test_dl)
    trainer.get_metrics()[1].report_best()

    save_filepath = logdir_abspath(save_filename)
    if save_filepath:
        utils.save(model, save_filepath)
    else:
        logger.warn('Model is not saved as save file is not in config')







