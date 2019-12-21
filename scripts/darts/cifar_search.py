import yaml
import os

from FastAutoAugment.darts.darts_arch_trainer import DartsArchTrainer
from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.search_arch import search_arch
from FastAutoAugment.darts.darts_dag_mutator import DartsDagMutator

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_search')

    conf_common = conf['common']
    logdir = conf_common['logdir']
    conf_data = conf['dataset']
    conf_search = conf['darts']['search']

    dag_mutator = DartsDagMutator()
    arch_trainer = DartsArchTrainer()
    found_model_desc = search_arch(conf_common, conf_data, conf_search,
                                   dag_mutator, arch_trainer)

    # log the model we found
    found_model_yaml = yaml.dump(found_model_desc)
    with open(os.path.join(logdir, 'model_desc.yaml'), 'w') as f:
        f.write(found_model_yaml)
