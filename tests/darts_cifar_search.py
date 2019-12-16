from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts.search import search_arch
from FastAutoAugment.darts.model_desc import darts_model_desc

import yaml

if __name__ == '__main__':
    conf = common_init(config_filepath=None,
        defaults_filepath='confs/defaults.yaml', experiment_name='cifar_search')

    # region conf vars
    conf_search     = conf['darts']['search']
    conf_ds         = conf['dataset']
    ch_in           = conf_ds['ch_in']
    n_classes       = conf_ds['n_classes']
    init_ch_out     = conf_search['init_ch_out']
    n_cells         = conf_search['n_cells']
    n_nodes         = conf_search['n_nodes']
    n_out_nodes     = conf_search['n_out_nodes']
    stem_multiplier = conf_search['stem_multiplier']
    # endregion

    model_desc = darts_model_desc(ch_in, n_classes, n_cells, n_nodes, n_out_nodes,
                     init_ch_out, stem_multiplier, True)

    found_model_desc = search_arch(conf, model_desc)

    found_model_yaml = yaml.dump(found_model_desc)
    with open(os.path.join(logdir, 'found_arch.yaml'), 'w') as f:
        f.write(found_model_yaml)
