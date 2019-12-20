from FastAutoAugment.nas.model_desc import RunMode
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.search import search_arch
from FastAutoAugment.nas.model_desc_builder import ModelDescBuilder
from FastAutoAugment.darts.darts_strategy import DartsStrategy

import yaml
import os

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_search')

    conf_common = conf['common']
    logdir = conf_common['logdir']
    conf_data = conf['dataset']
    conf_search = conf['darts']['search']

    strategy = DartsStrategy()
    found_model_desc = search_arch(conf_common, conf_data, conf_search, strategy)

    found_model_yaml = yaml.dump(found_model_desc)
    with open(os.path.join(logdir, 'model_desc.yaml'), 'w') as f:
        f.write(found_model_yaml)
