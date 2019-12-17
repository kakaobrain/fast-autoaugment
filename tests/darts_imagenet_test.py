from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts.test import test_arch

import os

import yaml

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/imagenet_darts.yaml',
        defaults_filepath='confs/defaults.yaml')

    conf_ds         = conf['dataset']
    conf_test       = conf['darts']['test']
    model_desc_file = conf_test['model_desc_file']
    conf_model_desc   = conf_test['model_desc']
    logdir          = conf['logdir']

    with open(os.path.join(logdir, model_desc_file), 'r') as f:
        model_desc = yaml.load(f)

    best_top1, model = test_arch(conf, model_desc)