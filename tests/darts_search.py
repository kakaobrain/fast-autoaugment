from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts.search import search

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/cifar_darts_search.yaml',
        defaults_filepath='confs/defaults.yaml')

    search(conf)