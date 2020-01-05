from FastAutoAugment.common.common import common_init
from FastAutoAugment.darts.darts_micro_builder import DartsMicroBuilder
from FastAutoAugment.nas.evaluate import eval_arch

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/imagenet_darts.yaml',
                       experiment_name='imagenet_eval')

    conf_eval = conf['nas']['eval']

    micro_builder = DartsMicroBuilder()
    eval_arch(conf_eval, micro_builder=micro_builder)

    exit(0)

