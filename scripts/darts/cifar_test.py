import os
import yaml

from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.test_arch import test_arch

if __name__ == '__main__':
    conf = common_init(defaults_filepath='confs/defaults.yaml',
                       experiment_name='cifar_test')

    conf_common = conf['common']
    conf_data         = conf['dataset']
    conf_test       = conf['darts']['test']

    best_top1, model = test_arch(conf_common, conf_data, conf_test)

    exit(0)

