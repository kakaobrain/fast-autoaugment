from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts.search import search_arch

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_search')

    search_arch(conf)