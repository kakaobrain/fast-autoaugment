import os
import yaml

from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.evaluate import eval_arch

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/darts_cifar.yaml',
                       experiment_name='cifar_eval')

    conf_common = conf['common']
    conf_data = conf['dataset']
    conf_test = conf['nas']['eval']

    best_top1, model = eval_arch(conf_common, conf_data, conf_test)

    exit(0)

