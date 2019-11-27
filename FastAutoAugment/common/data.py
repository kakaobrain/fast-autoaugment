import logging
import os
import sys

import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit

from .aug_policies import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, \
    fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
from .augmentations import *
from .common import get_logger
from .imagenet import ImageNet


def get_datasets(dataset, dataroot, transform_train, transform_test):
    total_trainset, testset = None, None

    if dataset == 'cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar10':
        total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=True, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'reduced_svhn':
        total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=True, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-1000, random_state=0)  # 1000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        targets = [total_trainset.targets[idx] for idx in train_idx]
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
        idx120 = [904, 385, 759, 884, 784, 844, 132, 214, 990, 786, 979, 582, 104, 288, 697, 480, 66, 943, 308, 282, 118, 926, 882, 478, 133, 884, 570, 964, 825, 656, 661, 289, 385, 448, 705, 609, 955, 5, 703, 713, 695, 811, 958, 147, 6, 3, 59, 354, 315, 514, 741, 525, 685, 673, 657, 267, 575, 501, 30, 455, 905, 860, 355, 911, 24, 708, 346, 195, 660, 528, 330, 511, 439, 150, 988, 940, 236, 803, 741, 295, 111, 520, 856, 248, 203, 147, 625, 589, 708, 201, 712, 630, 630, 367, 273, 931, 960, 274, 112, 239, 463, 355, 955, 525, 404, 59, 981, 725, 90, 782, 604, 323, 418, 35, 95, 97, 193, 690, 869, 172]
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 500000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)

        # filter out
        train_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, train_idx))
        valid_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, valid_idx))
        test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))

        targets = [idx120.index(total_trainset.targets[idx]) for idx in train_idx]
        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        for idx in range(len(testset.samples)):
            if testset.samples[idx][1] not in idx120:
                continue
            testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    return  total_trainset, testset

# target_lb allows to filter dataset for a specific class, used only for experimentation
def get_train_sampler(cv_ratio:float, cv_fold:int, total_trainset, horovod, target_lb:int):
    """Splits train set into train and validation sets using stratified random sampling.

    Arguments:
        cv_ratio {float} -- % of data to put in test set
        cv_fold {int} -- Total of 5 folds are created, cv_fold specifies which one to use
        target_lb {int} -- If >= 0 then trainset is filtered for only that target class ID
    """
    assert cv_fold >= 0

    train_sampler, valid_sampler = None, None
    if cv_ratio > 0.0: # if cv_ratio is not specified then val sampler will be empty
        # stratified shuffle cv_ratio will yield return total of n_splits, each cv_ratio containing
        # tuple of train and test set with test set size portion = cv_ratio, while samples for
        # each class having same proportions as original dataset
        # TODO: random_state should be None so np.random is used
        # TODO: keep hardcoded n_splits=5?
        sss = StratifiedShuffleSplit(n_splits=5, test_size=cv_ratio, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)

        # we have 5 plits, but will select only one of them as specified by cv_fold
        for _ in range(cv_fold + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

        if horovod:
            import horovod.torch as hvd
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        # this means no sampling, validation set would be empty
        valid_sampler = SubsetSampler([])

        if horovod:
            import horovod.torch as hvd
            train_sampler = torch.utils.data.distributed.DistributedSampler(valid_sampler, num_replicas=hvd.size(), rank=hvd.rank())

    return train_sampler, valid_sampler


def add_augs(transform_train, aug:str, cutout:int):
    logger = get_logger('Fast AutoAugment')
    logger.setLevel(logging.INFO)

    # TODO: total_aug remains None in original code
    total_aug = augs = None

    if isinstance(aug, list):
        logger.debug('augmentation provided.')
        transform_train.transforms.insert(0, Augmentation(aug))
    elif aug:
        logger.debug('augmentation: %s' % aug)
        if aug == 'fa_reduced_cifar10':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

        elif aug == 'fa_reduced_imagenet':
            transform_train.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))

        elif aug == 'fa_reduced_svhn':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_svhn()))

        elif aug == 'arsaug':
            transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
        elif aug == 'autoaug_cifar10':
            transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif aug == 'autoaug_extend':
            transform_train.transforms.insert(0, Augmentation(autoaug_policy()))
        elif aug in ['default', 'inception', 'inception320']:
            pass
        else:
            raise ValueError('not found augmentations. %s' % aug)

    # add cutout transform
    if cutout > 0:
        transform_train.transforms.append(CutoutDefault(cutout))

    #else returns augs = None
    return total_aug, augs

def get_transforms(dataset):
    transform_train, transform_test = None, None

    if 'cifar' in dataset or 'svhn' in dataset:
        _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        _IMAGENET_PCA = {
            'eigval': [0.2175, 0.0188, 0.0045],
            'eigvec': [
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ]
        }

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
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

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    return transform_train, transform_test

def get_dataloaders(dataset, batch, dataroot, aug=None, cutout=0, cv_ratio=0.15,
    cv_fold=0, horovod=False, target_lb=-1, num_workers=None):

    if num_workers is None:
        if 'pydevd' in sys.modules: # if debugging
            num_workers = 0
        else:
            num_workers = 32

    # get usual random crop/flip transforms
    transform_train, transform_test = get_transforms(dataset)

    # add additional transforms from config
    total_aug, augs = add_augs(transform_train, aug, cutout)

    total_trainset, testset = get_datasets(dataset, dataroot, transform_train, transform_test)

    # TODO: below will never get executed, set_preaug does not exist in PyTorch
    # if total_aug is not None and augs is not None:
    #     total_trainset.set_preaug(augs, total_aug)
    #     print('set_preaug-')

    train_sampler, valid_sampler = get_train_sampler(cv_ratio, cv_fold, total_trainset, horovod, target_lb)

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=num_workers/2, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True,
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