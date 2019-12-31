import os
import yaml

from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.eval_arch import eval_arch


if __name__ == '__main__':
    conf = common_init(defaults_filepath='confs/defaults.yaml',
                       experiment_name='cifar_test')

    conf_common = conf['common']
    conf_data         = conf['dataset']
    conf_eval       = conf['nas']['test']

    best_top1, model = eval_arch(conf_common, conf_data, conf_eval)

