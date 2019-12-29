from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.eval_arch import eval_arch

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/imagenet_darts.yaml',
                       experiment_name='imagenet_eval')

    conf_common = conf['common']
    conf_data = conf['dataset']
    conf_test = conf['nas']['test']

    best_top1, model = eval_arch(conf_common, conf_data, conf_test)
