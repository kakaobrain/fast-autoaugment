import torch

from FastAutoAugment.nas.bilevel_arch_trainer import BilevelArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.search_arch import create_model, \
                                            get_data, save_found_model_desc
from FastAutoAugment.random_arch.random_dag_mutator import RandomDagMutator

from FastAutoAugment.nas.test_arch import test_arch

from FastAutoAugment.nas.search_arch import _create_model_desc

if __name__ == '__main__':
    conf = common_init(defaults_filepath='confs/cifar_random.yaml',
                       experiment_name='cifar_random_search')

    # region config
    conf_common = conf['common']
    conf_data = conf['dataset']
    conf_test = conf['random']['test']
    # endregion

    device = torch.device('cuda')

    # create model and save it to yaml
    # NOTE: there is no search here as
    # the models are just randomly sampled
    dag_mutator = RandomDagMutator()
    model = create_model(conf_data, conf_test, dag_mutator, device)
    model_desc = _create_model_desc(conf_data, conf_test['model'], dag_mutator)
    save_found_model_desc(conf_common, conf_test, model_desc)

    # train it fully
    best_top1, model = test_arch(conf_common, conf_data, conf_test)

    exit(0)
