from FastAutoAugment.nas.arch_trainer import ArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas import search
from FastAutoAugment.petridish.petridish_mutator import PetridishMutator


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                       experiment_name='cifar_search')

    # region config
    conf_search = conf['nas']['search']
    # endregion

    dag_mutator = PetridishMutator()
    trainer_class = ArchTrainer

    search.search_arch(conf_search, dag_mutator, trainer_class)

    exit(0)
