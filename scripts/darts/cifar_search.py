import torch

from FastAutoAugment.nas.bilevel_arch_trainer import BilevelArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas import search
from FastAutoAugment.darts.darts_dag_mutator import DartsDagMutator

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/darts_cifar.yaml',
                       experiment_name='cifar_search')

    # region config
    conf_common = conf['common']
    conf_data = conf['dataset']
    conf_search = conf['nas']['search']
    conf_loader = conf_search['loader']
    conf_train = conf_search['trainer']
    # endregion

    device = torch.device(conf_common['device'])

    # create model
    dag_mutator = DartsDagMutator()
    model = search.create_model(conf_data, conf_search, dag_mutator, device)

    # get data
    train_dl, val_dl = search.get_data(conf_common, conf_data, conf_loader)

    # search arch
    arch_trainer = BilevelArchTrainer(conf_train, model, device)
    arch_trainer.fit(train_dl, val_dl)
    found_model_desc = arch_trainer.get_model_desc()

    # save found model
    search.save_found_model_desc(conf_common, conf_search, found_model_desc)

    exit(0)
