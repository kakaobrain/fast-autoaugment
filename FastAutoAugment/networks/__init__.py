import torch
from pretrainedmodels import models

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from FastAutoAugment.networks.pyramidnet import PyramidNet
from FastAutoAugment.networks.shakeshake.shake_resnet import ShakeResNet
from FastAutoAugment.networks.wideresnet import WideResNet
from FastAutoAugment.networks.preresnet.preresnet import preresnet200


def get_model(conf, num_class=10, data_parallel=True):
    name = conf['type']

    if name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)

    elif name == 'pyramid':
        model = PyramidNet('cifar10', depth=conf['depth'], alpha=conf['alpha'], num_classes=num_class, bottleneck=conf['bottleneck'])

    elif name == 'resnet50':
        model = models.resnet50(num_classes=num_class, pretrained=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(512 * 1, num_class)
    elif name == 'resnet200':
        model = preresnet200(num_classes=num_class)
        model.features._modules['final_pool'] = nn.AdaptiveAvgPool2d((1, 1))

    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        import horovod.torch as hvd
        device = torch.device('cuda', hvd.local_rank())
        model = model.to(device)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
