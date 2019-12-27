import torch

from FastAutoAugment.nas.bilevel_arch_trainer import BilevelArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.search_arch import create_model, \
                                            get_data, save_found_model_desc
from FastAutoAugment.darts.darts_dag_mutator import DartsDagMutator

if __name__ == '__main__':
    conf = common_init(defaults_filepath='confs/defaults.yaml',
                       experiment_name='cifar_search')

    # region config
    conf_common = conf['common']
    conf_data = conf['dataset']
    conf_search = conf['darts']['search']
    conf_loader = conf_search['loader']
    epochs = conf_loader['epochs']
    # endregion

    device = torch.device('cuda')

    # create model
    dag_mutator = DartsDagMutator()
    model = create_model(conf_data, conf_search, dag_mutator, device)

    # get data
    train_dl, val_dl = get_data(conf_common, conf_loader, conf_data)

    # search arch
    arch_trainer = BilevelArchTrainer(conf_common, conf_search, model, device)
    found_model_desc, *_ = arch_trainer.fit(train_dl, val_dl, epochs)

    # save found model
    save_found_model_desc(conf_common, conf_search, found_model_desc)

    exit(0)
