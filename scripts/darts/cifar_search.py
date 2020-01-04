from FastAutoAugment.darts.bilevel_arch_trainer import BilevelArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas import search
from FastAutoAugment.darts.darts_micro_builder import DartsMicroBuilder


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/darts_cifar.yaml',
                       experiment_name='cifar_search')

    # region config
    conf_search = conf['nas']['search']
    # endregion

    micro_builder = DartsMicroBuilder()
    trainer_class = BilevelArchTrainer

    search.search_arch(conf_search, micro_builder, trainer_class)

    exit(0)
