from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.test_arch import test_arch
from FastAutoAugment.nas.model_desc_builder import ModelDescBuilder
from FastAutoAugment.nas.model_desc import RunMode

import os

import yaml

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/imagenet_darts.yaml',
        defaults_filepath='confs/defaults.yaml')

    conf_data         = conf['dataset']
    conf_test       = conf['darts']['test']
    model_desc_file = conf_test['model_desc_file']
    conf_model_desc = conf_test['model_desc']
    logdir          = conf['logdir']

    with open(os.path.join(logdir, model_desc_file), 'r') as f:
        found_model_desc = yaml.load(f, Loader=yaml.Loader)

    builder = ModelDescBuilder(conf_data, conf_model_desc,
                               run_mode=RunMode.EvalTrain, template=found_model_desc)
    model_desc = builder.get_model_desc()

    best_top1, model = test_arch(conf, model_desc)