import json
import os
from FastAutoAugment.common.common import get_logger, common_init
from FastAutoAugment.data_aug.train import train_and_eval


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/wresnet40x2_cifar10_b512.yaml',
        defaults_filepath="confs/defaults.yaml", param_args=["--aug", "fa_reduced_cifar10"])
    logger = get_logger()

    import time
    t = time.time()
    save_path = os.path.join(conf['logdir'], 'model.pth')
    result = train_and_eval(conf, val_ratio=conf['val_ratio'], val_fold=conf['val_fold'],
                            save_path=save_path, only_eval=conf['only_eval'], metric='test')
    elapsed = time.time() - t

    logger.info('training done.')
    logger.info('model: %s' % conf['model'])
    logger.info('augmentation: %s' % conf['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info('Save path: %s' % save_path)
