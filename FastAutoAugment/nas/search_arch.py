import os
from typing import Tuple

import torch
from torch.utils.data.dataloader import DataLoader

from ..common.common import get_logger
from ..common.config import Config
from .model import Model
from ..common.data import get_dataloaders
from .model_desc import ModelDesc, RunMode
from .model_desc_builder import ModelDescBuilder
from .dag_mutator import DagMutator
from .arch_trainer import ArchTrainer


def save_found_model_desc(conf_common: Config, conf_search: Config,
                          found_model_desc:ModelDesc):
    logger = get_logger()
    model_desc_file = conf_search['model_desc_file']
    logdir = conf_common['logdir']

    if model_desc_file and logdir:
        model_desc_save_path = os.path.join(logdir, model_desc_file)
        with open(model_desc_save_path, 'w') as f:
            f.write(found_model_desc.serialize())
        logger.info(f"Best architecture saved in {model_desc_save_path}")
    else:
        logger.info(f"Best architecture is not saved because file path config not set")

def get_data(conf_common:Config, conf_loader:Config, conf_data:Config)\
        -> Tuple[DataLoader, DataLoader]:
    # region conf vars
    horovod = conf_common['horovod']
    # dataset
    ds_name = conf_data['name']
    max_batches = conf_data['max_batches']
    dataroot = conf_data['dataroot']
    # data loader
    aug = conf_loader['aug']
    cutout = conf_loader['cutout']
    val_ratio = conf_loader['val_ratio']
    batch_size = conf_loader['batch']
    val_fold = conf_loader['val_fold']
    n_workers = conf_loader['n_workers']
    # endregion

    train_dl, val_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=True, load_test=False,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod,
        n_workers=n_workers, max_batches=max_batches)

    return train_dl, val_dl

def _create_model_desc(conf_data: Config, conf_model_desc: Config,
                      dag_mutator: DagMutator) -> ModelDesc:
    builder = ModelDescBuilder(
        conf_data, conf_model_desc, run_mode=RunMode.Search)
    model_desc = builder.get_model_desc()
    dag_mutator.mutate(model_desc)
    return model_desc

def create_model(conf_data: Config, conf_search: Config,
                 dag_mutator: DagMutator, device) -> Model:
    conf_model_desc = conf_search['model_desc']
    model_desc = _create_model_desc(conf_data, conf_model_desc, dag_mutator)
    model = Model(model_desc)
    # if data_parallel:
    #     model = nn.DataParallel(model).to(device)
    # else:
    # TODO: enable DataParallel
    model = model.to(device)
    return model
