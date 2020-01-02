from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.evaluate import eval_arch
from FastAutoAugment.petridish.petridish_mutator import PetridishMutator

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                       experiment_name='cifar_eval')

    conf_eval = conf['nas']['eval']

    # TODO: we need better name for mutator class
    # currently this is needed to register petridish op
    dag_mutator = PetridishMutator()

    eval_arch(conf_eval)

    exit(0)

