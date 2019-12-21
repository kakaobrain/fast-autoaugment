from typing import Tuple, Any, Iterable, Optional
import  torch.nn as nn
import torch
import os
import yaml

from ..common.config import Config
from .model import Model
from ..common.data import get_dataloaders
from ..common.common import get_logger, get_tb_writer
from ..common.optimizer import get_lr_scheduler, get_optimizer, get_lossfn
from .vis_genotype import draw_model_desc
from ..common.train_test_utils import train_test
from .model_desc import ModelDesc, RunMode
from .model_desc_builder import ModelDescBuilder
from .dag_mutator import DagMutator
from .arch_trainer import ArchTrainer

def search_arch(conf_common:Config, conf_data:Config, conf_search:Config,
                dag_mutator:DagMutator, arch_trainer:ArchTrainer)->ModelDesc:
    logger = get_logger()

    # region conf vars
    horovod       = conf_common['horovod']
    report_freq   = conf_common['report_freq']
    plotsdir      = conf_common['plotsdir']
    chkptdir      = conf_common['chkptdir']
    # dataset
    ds_name       = conf_data['name']
    max_batches   = conf_data['max_batches']
    dataroot      = conf_data['dataroot']
    # data loader
    conf_loader   = conf_search['loader']
    aug           = conf_loader['aug']
    cutout        = conf_loader['cutout']
    val_ratio     = conf_loader['val_ratio']
    batch_size    = conf_loader['batch']
    epochs        = conf_loader['epochs']
    val_fold      = conf_loader['val_fold']
    n_workers     = conf_loader['n_workers']
    # search
    data_parallel = conf_search['data_parallel']
    conf_model_desc = conf_search['model_desc']
    # endregion

    builder = ModelDescBuilder(conf_data, conf_model_desc, run_mode=RunMode.Search)
    model_desc = builder.get_model_desc()
    dag_mutator.mutate(model_desc)

    # get data
    train_dl, val_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=True, load_test=False,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod,
        n_workers=n_workers, max_batches=max_batches)

    device = torch.device('cuda')

    model = Model(model_desc)
    # if data_parallel:
    #     model = nn.DataParallel(model).to(device)
    # else:
    # TODO: enable DataParallel
    model = model.to(device)

    found_model_desc, *_ = arch_trainer.fit(conf_search, train_dl, val_dl,
                            model, epochs, plotsdir, report_freq)

    logger.info("Best architecture\n{}".format(yaml.dump(found_model_desc)))
    return found_model_desc


