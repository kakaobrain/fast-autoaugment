from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.evaluate import eval_arch

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                       experiment_name='cifar_eval')

    conf_eval = conf['nas']['eval']
    eval_arch(conf_eval)

    exit(0)

