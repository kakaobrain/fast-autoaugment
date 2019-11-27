from FastAutoAugment.common import common_init
from FastAutoAugment.darts_search import darts_search

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/cifar_darts.yaml',
        defaults_filepath='confs/cifar_darts.yaml')
    darts_search(conf)