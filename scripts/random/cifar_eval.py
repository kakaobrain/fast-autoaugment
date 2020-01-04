from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.evaluate import eval_arch
from FastAutoAugment.random_arch.random_dag_mutator import RandomDagMutator
from FastAutoAugment.nas.model_desc import RunMode

from FastAutoAugment.nas import nas_utils

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/random_cifar.yaml',
                       experiment_name='cifar_random_search')

    # region config
    conf_search = conf['nas']['search']
    conf_eval = conf['nas']['eval']
    conf_model_desc = conf_search['model_desc']
    model_desc_filename = conf_eval['model_desc_file']
    # endregion

    # create model and save it to yaml
    # NOTE: there is no search here as the models are just randomly sampled
    model_desc = nas_utils.create_model_desc(conf_model_desc,
                                             run_mode=RunMode.Search,
                                             dag_mutator=RandomDagMutator())

    # save model to location specified by eval config
    nas_utils.save_model_desc(model_desc_filename, model_desc)

    # evaluate architecture using eval settings
    eval_arch(conf_eval)

    exit(0)
