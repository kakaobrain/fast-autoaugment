from FastAutoAugment.nas.bilevel_arch_trainer import BilevelArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas import search
from FastAutoAugment.darts.darts_dag_mutator import DartsDagMutator


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/darts_cifar.yaml',
                       experiment_name='cifar_search')

    # region config
    conf_search = conf['nas']['search']
    # endregion

    dag_mutator = DartsDagMutator()
    trainer_class = BilevelArchTrainer

    search.search_arch(conf_search, dag_mutator, trainer_class)

    exit(0)
