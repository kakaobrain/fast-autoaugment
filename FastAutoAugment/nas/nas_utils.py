from typing import Tuple, Optional
import os

from torch.utils.data.dataloader import DataLoader

from .model_desc import RunMode, ModelDesc
from .model_desc_builder import ModelDescBuilder
from .dag_mutator import DagMutator
from ..common.config import Config
from .model import Model
from ..common.data import get_dataloaders
from ..common.common import get_logger, logdir_abspath

def create_model(conf_model_desc: Config, device, run_mode:RunMode,
                 dag_mutator: Optional[DagMutator]=None,
                 template_model_desc:Optional[ModelDesc]=None) -> Model:
    builder = ModelDescBuilder(conf_model_desc,
                               run_mode=RunMode.Search,
                               template=template_model_desc)
    model_desc = builder.get_model_desc()
    if dag_mutator:
        dag_mutator.mutate(model_desc)

    model = Model(model_desc)
    # TODO: enable DataParallel
    # if data_parallel:
    #     model = nn.DataParallel(model).to(device)
    # else:
    model = model.to(device)
    return model


def get_train_test_data(conf_loader:Config)\
        -> Tuple[DataLoader, Optional[DataLoader]]:
    # region conf vars
    # dataset
    conf_data = conf_loader['dataset']
    ds_name = conf_data['name']
    max_batches = conf_data['max_batches']
    dataroot = conf_data['dataroot']

    aug = conf_loader['aug']
    cutout = conf_loader['cutout']
    val_ratio = conf_loader['val_ratio']
    batch_size = conf_loader['batch']
    val_fold = conf_loader['val_fold']
    n_workers = conf_loader['n_workers']
    horovod = conf_loader['horovod']
    load_train = conf_loader['load_train']
    load_test = conf_loader['load_test']
    # endregion

    train_dl, val_dl, *_ = get_dataloaders(
        ds_name, batch_size, dataroot,
        aug=aug, cutout=cutout, load_train=load_train, load_test=load_test,
        val_ratio=val_ratio, val_fold=val_fold, horovod=horovod,
        n_workers=n_workers, max_batches=max_batches)
    assert train_dl is not None
    return train_dl, val_dl
