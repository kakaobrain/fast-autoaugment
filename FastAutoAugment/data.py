import logging
import os

import torch
import torchvision

from torch.utils.data import SubsetRandomSampler, Sampler, Subset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from FastAutoAugment.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, random_search2048, \
    fa_reduced_imagenet, fa_reduced_cifar10
# fa_wresnet40x2, fa_wresnet28x10, fa_pyramid_e300, fa_pyramid_c100, fa_wresnet40x2_c100, \
# fa_resnet50_imagenet_minusloss, fa_wresnet40x2_cifar100, fa_wresnet40x2_cifar100_r5, fa_wresnet40x2_cifar10, \
# fa_wresnet28x10_cifar100, fa_wresnet28x10_cifar10, fa_pyramid_cifar10, fa_pyramid_cifar100, \
# fa_resnet50_rimagenet, fa_shake26_2x96d_cifar100, fa_shake26_2x96d_cifar10, fa_reduced_cifar10_progressive
from FastAutoAugment.augmentations import *
from FastAutoAugment.common import get_logger

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def get_dataloaders(dataset, batch, dataroot, split=0.0, split_idx=0, horovod=False):
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif 'imagenet' in dataset:
        if C.get()['aug'] in ['inception', 'fa_reduced_rimagenet'] or isinstance(C.get()['aug'], (list, tuple, dict)):
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if C.get()['model']['type'] == 'resnet200':
                # Instead, we test a single 320×320 crop from s = 320
                transform_test = transforms.Compose([
                    transforms.Resize(320),
                    transforms.CenterCrop(320),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform_test = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError(C.get()['aug'])
    else:
        raise ValueError('dataset=%s' % dataset)

    if isinstance(C.get()['aug'], list):
        logger.debug('augmentation provided.')
        transform_train.transforms.insert(0, Augmentation(C.get()['aug']))
    else:
        logger.debug('augmentation: %s' % C.get()['aug'])
        if C.get()['aug'] == 'random2048':
            transform_train.transforms.insert(0, Augmentation(random_search2048()))
        elif C.get()['aug'] == 'fa_reduced_cifar10':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif C.get()['aug'] == 'fa_reduced_imagenet':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_imagenet()))

        elif C.get()['aug'] == 'arsaug':
            transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
        elif C.get()['aug'] == 'autoaug_cifar10':
            transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif C.get()['aug'] == 'autoaug_extend':
            transform_train.transforms.insert(0, Augmentation(autoaug_policy()))
        elif C.get()['aug'] in ['default', 'inception', 'inception320']:
            pass
        else:
            raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.train_labels)
        train_idx, valid_idx = next(sss)
        train_labels = [total_trainset.train_labels[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.train_labels = train_labels

        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'imagenet/train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'imagenet/val'), transform=transform_test)

        # compatibility
        total_trainset.train_labels = [lb for _, lb in total_trainset.samples]
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
        idx120 = [904, 385, 759, 884, 784, 844, 132, 214, 990, 786, 979, 582, 104, 288, 697, 480, 66, 943, 308, 282, 118, 926, 882, 478, 133, 884, 570, 964, 825, 656, 661, 289, 385, 448, 705, 609, 955, 5, 703, 713, 695, 811, 958, 147, 6, 3, 59, 354, 315, 514, 741, 525, 685, 673, 657, 267, 575, 501, 30, 455, 905, 860, 355, 911, 24, 708, 346, 195, 660, 528, 330, 511, 439, 150, 988, 940, 236, 803, 741, 295, 111, 520, 856, 248, 203, 147, 625, 589, 708, 201, 712, 630, 630, 367, 273, 931, 960, 274, 112, 239, 463, 355, 955, 525, 404, 59, 981, 725, 90, 782, 604, 323, 418, 35, 95, 97, 193, 690, 869, 172]
        total_trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'imagenet/train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'imagenet/val'), transform=transform_test)

        # compatibility
        total_trainset.train_labels = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 500000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.train_labels)
        train_idx, valid_idx = next(sss)

        # filter out
        train_idx = list(filter(lambda x: total_trainset.train_labels[x] in idx120, train_idx))
        valid_idx = list(filter(lambda x: total_trainset.train_labels[x] in idx120, valid_idx))
        test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))

        train_labels = [idx120.index(total_trainset.train_labels[idx]) for idx in train_idx]
        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.train_labels = train_labels

        for idx in range(len(testset.samples)):
            if testset.samples[idx][1] not in idx120:
                continue
            testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.train_labels)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

        if horovod:
            import horovod.torch as hvd
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        train_sampler = None
        valid_sampler = SubsetSampler(list(range(16)))

        if horovod:
            import horovod.torch as hvd
            train_sampler = torch.utils.data.distributed.DistributedSampler(total_trainset, num_replicas=hvd.size(), rank=hvd.rank())

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=32, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=16, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
