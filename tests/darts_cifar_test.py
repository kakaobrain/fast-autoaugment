from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts.test import test_arch

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_test')

    test_arch(conf)
