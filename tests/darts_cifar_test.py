from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.test import test_arch
from FastAutoAugment.nas.model_desc_builder import ModelDescBuilder
from FastAutoAugment.nas.model_desc import RunMode

import os

import yaml

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_test')

    conf_ds         = conf['dataset']
    conf_test       = conf['darts']['test']
    model_desc_file = conf_test['model_desc_file']
    conf_model_desc = conf_test['model_desc']
    logdir          = conf['logdir']

    with open(os.path.join(logdir, model_desc_file), 'r') as f:
        found_model_desc = yaml.safe_load(f)

    builder = ModelDescBuilder(conf_ds, conf_model_desc,
                                run_mode=RunMode.EvalTrain, template=found_model_desc)
    model_desc = builder.get_model_desc()

    best_top1, model = test_arch(conf, model_desc)

