from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.test_arch import test_arch

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/imagenet_darts.yaml',
                       defaults_filepath='confs/defaults.yaml',
                       experiment_name='imagenet_test')

    conf_common = conf['common']
    conf_data = conf['dataset']
    conf_test = conf['darts']['test']

    best_top1, model = test_arch(conf_common, conf_data, conf_test)
