from collections import defaultdict
from FastAutoAugment.common import common_init
from FastAutoAugment.aug_search import search_aug

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/wresnet40x2_cifar10_b512.yaml',
        defaults_filepath="confs/defaults.yaml")
    search_aug(conf)

