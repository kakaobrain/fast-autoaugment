import torch
import os

from ..common.common import get_logger
from ..common.config import Config
from .model import Model
from ..common.data import get_dataloaders
from .model_desc import ModelDesc, RunMode
from .model_desc_builder import ModelDescBuilder
from .dag_mutator import DagMutator
from .arch_trainer import ArchTrainer


def search_arch(conf_common: Config, conf_data: Config, conf_search: Config,
                dag_mutator: DagMutator, arch_trainer: ArchTrainer) -> ModelDesc:
    logger = get_logger()

    # region conf vars
    horovod = conf_common['horovod']
    logger_freq = conf_common['logger_freq']
    plotsdir = conf_common['plotsdir']
    logdir = conf_common['logdir']
    # dataset
    ds_name = conf_data['name']
    max_batches = conf_data['max_batches']
    dataroot = conf_data['dataroot']
    # data loader
    conf_loader = conf_search['loader']
    aug = conf_loader['aug']
    cutout = conf_loader['cutout']
    val_ratio = conf_loader['val_ratio']
    batch_size = conf_loader['batch']
    epochs = conf_loader['epochs']
    val_fold = conf_loader['val_fold']
    n_workers = conf_loader['n_workers']
    # search
    model_desc_file = conf_search['model_desc_file']
    conf_model_desc = conf_search['model_desc']
    # endregion

    device = torch.device('cuda')
    model = create_model(conf_data, conf_model_desc, dag_mutator, device)

    # get data
    train_dl, val_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=True, load_test=False,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod,
        n_workers=n_workers, max_batches=max_batches)

    found_model_desc, *_ = arch_trainer.fit(conf_search, model, device,
                                            train_dl, val_dl,
                                            epochs, plotsdir, logger_freq)

    if model_desc_file and logdir:
        model_desc_save_path = os.path.join(logdir, model_desc_file)
        with open(model_desc_save_path, 'w') as f:
            f.write(found_model_desc.serialize())
        logger.info(f"Best architecture saved in {model_desc_save_path}")

    return found_model_desc


def create_model_desc(conf_data: Config, conf_model_desc: Config,
                      dag_mutator: DagMutator) -> ModelDesc:
    builder = ModelDescBuilder(
        conf_data, conf_model_desc, run_mode=RunMode.Search)
    model_desc = builder.get_model_desc()
    dag_mutator.mutate(model_desc)
    return model_desc

def create_model(conf_data: Config, conf_model_desc: Config,
                 dag_mutator: DagMutator, device) -> Model:
    model_desc = create_model_desc(conf_data, conf_model_desc, dag_mutator)
    model = Model(model_desc)
    # if data_parallel:
    #     model = nn.DataParallel(model).to(device)
    # else:
    # TODO: enable DataParallel
    model = model.to(device)
    return model
