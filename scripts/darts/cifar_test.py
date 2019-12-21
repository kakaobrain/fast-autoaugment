import os
import yaml

from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.test_arch import test_arch
from FastAutoAugment.nas.model_desc_builder import ModelDescBuilder
from FastAutoAugment.nas.model_desc import RunMode, ModelDesc

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_test')

    conf_common = conf['common']
    conf_data         = conf['dataset']
    logdir          = conf_common['logdir']
    conf_test       = conf['darts']['test']
    model_desc_file = conf_test['model_desc_file']
    conf_model_desc = conf_test['model_desc']

    # open the model description we want to test
    with open(os.path.join(logdir, model_desc_file), 'r') as f:
        found_model_desc = yaml.load(f, Loader=yaml.Loader)

    # compile to PyTorch model
    builder = ModelDescBuilder(conf_data, conf_model_desc,
                                run_mode=RunMode.EvalTrain,
                                template=found_model_desc)
    model_desc = builder.get_model_desc()

    best_top1, model = test_arch(conf_common, conf_data, conf_test, model_desc)

