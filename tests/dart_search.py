from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts_search import search

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/cifar_darts.yaml',
        defaults_filepath='confs/cifar_darts.yaml')

    search(conf)