from torch.utils.data.dataloader import DataLoader
import os
import sys
from typing import Tuple, Union, Optional

import torch
import torchvision
from PIL import Image

from torch.utils.data import \
    SubsetRandomSampler, Sampler, Subset, ConcatDataset, Dataset, random_split
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit

from .aug_policies import arsaug_policy, autoaug_policy, autoaug_paper_cifar10,\
    fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
from .augmentations import *
from ..common.common import get_logger
from .imagenet import ImageNet

DatasetLike = Union[Dataset, Subset, ConcatDataset]


def get_dataloaders(dataset:str, batch_size, dataroot:str, aug, cutout:int,
    load_train:bool, load_test:bool, val_ratio:float, val_fold=0,
    horovod=False, target_lb=-1, n_workers:int=None, max_batches:int=-1) \
        -> Tuple[Optional[DataLoader], Optional[DataLoader],
                 Optional[DataLoader], Optional[Sampler]]:

    logger = get_logger()

    # if debugging in vscode, workers > 0 gets termination
    if 'pydevd' in sys.modules:
        n_workers = 0
        logger.warn('Debugger is detected, lower performance settings may be used.')
    else: # use simple heuristic to auto select number of workers
        n_workers = int(torch.cuda.device_count()*4 if n_workers is None \
            else n_workers)
    logger.info('n_workers = {}'.format(n_workers))

    # get usual random crop/flip transforms
    transform_train, transform_test = get_transforms(dataset)

    # add additional aug and cutout transformations
    _add_augs(transform_train, aug, cutout)

    trainset, testset = _get_datasets(dataset, dataroot,
        load_train, load_test, transform_train, transform_test)

    # TODO: below will never get executed, set_preaug does not exist in PyTorch
    # if total_aug is not None and augs is not None:
    #     trainset.set_preaug(augs, total_aug)
    #     logger.info('set_preaug-')

    trainloader, validloader, testloader, train_sampler = None, None, None, None

    if trainset:
        if max_batches >= 0:
            max_size = max_batches*batch_size
            logger.warn('Trainset trimmed to max_batches = {}'.format(max_size))
            trainset = LimitDataset(trainset, max_size)
        # sample validation set from trainset if cv_ration > 0
        train_sampler, valid_sampler = _get_train_sampler(val_ratio, val_fold,
            trainset, horovod, target_lb)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True if train_sampler is None else False,
            num_workers=n_workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)
        if train_sampler is not None:
            validloader = torch.utils.data.DataLoader(trainset,
                batch_size=batch_size, shuffle=False,
                num_workers=n_workers, pin_memory=True, #TODO: set n_workers per ratio?
                sampler=valid_sampler, drop_last=False)
        # else validloader is left as None
    if testset:
        if max_batches >= 0:
            max_size = max_batches*batch_size
            logger.warn('Testset trimmed to max_batches = {}'.format(max_size))
            testset = LimitDataset(testset, max_batches*batch_size)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=True,
            sampler=None, drop_last=False
    )

    assert val_ratio > 0.0 or validloader is None

    logger.info('Dataset batches: train={}, val={}, test={}'.format(
        len(trainloader) if trainloader is not None else 'None',
        len(validloader) if validloader is not None else 'None',
        len(testloader) if testloader is not None else 'None'))

    # we have to return train_sampler because of horovod
    return trainloader, validloader, testloader, train_sampler

def get_transforms(dataset):
    if 'imagenet' in dataset:
        return _get_imagenet_transforms()

    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'cifar100':
        MEAN = [0.507, 0.487, 0.441]
        STD = [0.267, 0.256, 0.276]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'svhn':
        MEAN = [0.4914, 0.4822, 0.4465]
        STD = [0.2023, 0.1994, 0.20100]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=0.1)
        ]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('dataset not recognized: {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    test_transform = transforms.Compose(normalize)

    return train_transform, test_transform

class CutoutDefault:
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

class Augmentation:
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

class LimitDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n
        if hasattr(dataset, 'targets'):
            self.targets = dataset.targets[:n]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


def _get_datasets(dataset, dataroot, load_train:bool, load_test:bool,
        transform_train, transform_test)\
            ->Tuple[DatasetLike, DatasetLike]:
    logger = get_logger()
    trainset, testset = None, None

    if dataset == 'cifar10':
        if load_train:
            # NOTE: train transforms will also be applied to validation set
            trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True,
                download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.CIFAR10(root=dataroot, train=False,
                download=True, transform=transform_test)
    elif dataset == 'mnist':
        if load_train:
            trainset = torchvision.datasets.MNIST(root=dataroot, train=True,
                download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.MNIST(root=dataroot, train=False,
                download=True, transform=transform_test)
    elif dataset == 'fashionmnist':
        if load_train:
            trainset = torchvision.datasets.FashionMNIST(root=dataroot,
                train=True, download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.FashionMNIST(root=dataroot,
                train=False, download=True, transform=transform_test)
    elif dataset == 'reduced_cifar10':
        if load_train:
            trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True,
                download=True, transform=transform_train)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=46000)   # 4000
            sss = sss.split(list(range(len(trainset))), trainset.targets)
            train_idx, valid_idx = next(sss)
            targets = [trainset.targets[idx] for idx in train_idx]
            trainset = Subset(trainset, train_idx)
            trainset.targets = targets
        if load_test:
            testset = torchvision.datasets.CIFAR10(root=dataroot, train=False,
                download=True, transform=transform_test)
    elif dataset == 'cifar100':
        if load_train:
            trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True,
                download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.CIFAR100(root=dataroot, train=False,
                download=True, transform=transform_test)
    elif dataset == 'svhn':
        if load_train:
            trainset = torchvision.datasets.SVHN(root=dataroot, split='train',
                download=True, transform=transform_train)
            extraset = torchvision.datasets.SVHN(root=dataroot, split='extra',
                download=True, transform=transform_train)
            trainset = ConcatDataset([trainset, extraset])
        if load_test:
            testset = torchvision.datasets.SVHN(root=dataroot, split='test',
                download=True, transform=transform_test)
    elif dataset == 'reduced_svhn':
        if load_train:
            trainset = torchvision.datasets.SVHN(root=dataroot, split='train',
                download=True, transform=transform_train)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-1000) #1000
            sss = sss.split(list(range(len(trainset))), trainset.targets)
            train_idx, valid_idx = next(sss)
            targets = [trainset.targets[idx] for idx in train_idx]
            trainset = Subset(trainset, train_idx)
            trainset.targets = targets
        if load_test:
            testset = torchvision.datasets.SVHN(root=dataroot, split='test',
                download=True, transform=transform_test)
    elif dataset == 'imagenet':
        if load_train:
            trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),
                transform=transform_train)
            # compatibility
            trainset.targets = [lb for _, lb in trainset.samples]
        if load_test:
            testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),
                split='val', transform=transform_test)
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
        idx120 = [904, 385, 759, 884, 784, 844, 132, 214, 990, 786, 979, 582,
        104, 288, 697, 480, 66, 943, 308, 282, 118, 926, 882, 478, 133, 884,
        570, 964, 825, 656, 661, 289, 385, 448, 705, 609, 955, 5, 703, 713, 695,
        811, 958, 147, 6, 3, 59, 354, 315, 514, 741, 525, 685, 673, 657, 267,
        575, 501, 30, 455, 905, 860, 355, 911, 24, 708, 346, 195, 660, 528, 330,
        511, 439, 150, 988, 940, 236, 803, 741, 295, 111, 520, 856, 248, 203,
        147, 625, 589, 708, 201, 712, 630, 630, 367, 273, 931, 960, 274, 112,
        239, 463, 355, 955, 525, 404, 59, 981, 725, 90, 782, 604, 323, 418, 35,
        95, 97, 193, 690, 869, 172]
        if load_train:
            trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),
                transform=transform_train)
            # compatibility
            trainset.targets = [lb for _, lb in trainset.samples]

            sss = StratifiedShuffleSplit(n_splits=1,
                test_size=len(trainset) - 500000, random_state=0)  # 4000
            sss = sss.split(list(range(len(trainset))), trainset.targets)
            train_idx, valid_idx = next(sss)

            # filter out
            train_idx = list(filter(lambda x: trainset.labels[x] in idx120, train_idx))
            valid_idx = list(filter(lambda x: trainset.labels[x] in idx120, valid_idx))

            targets = [idx120.index(trainset.targets[idx]) for idx in train_idx]
            for idx in range(len(trainset.samples)):
                if trainset.samples[idx][1] not in idx120:
                    continue
                trainset.samples[idx] = (trainset.samples[idx][0],
                    idx120.index(trainset.samples[idx][1]))
            trainset = Subset(trainset, train_idx)
            trainset.targets = targets
            logger.info('reduced_imagenet train={}'.format(len(trainset)))
        if load_test:
            testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'),
                split='val', transform=transform_test)
            test_idx = list(filter(lambda x: testset.samples[x][1] in \
                idx120, range(len(testset))))
            for idx in range(len(testset.samples)):
                if testset.samples[idx][1] not in idx120:
                    continue
                testset.samples[idx] = (testset.samples[idx][0],
                    idx120.index(testset.samples[idx][1]))
            testset = Subset(testset, test_idx)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    return  trainset, testset

# target_lb allows to filter dataset for a specific class, not used
def _get_train_sampler(val_ratio:float, val_fold:int, trainset, horovod,
        target_lb:int=-1)->Tuple[Optional[Sampler], Sampler]:
    """Splits train set into train, validation sets, stratified rand sampling.

    Arguments:
        val_ratio {float} -- % of data to put in valid set
        val_fold {int} -- Total of 5 folds are created, val_fold specifies which
            one to use
        target_lb {int} -- If >= 0 then trainset is filtered for only that
            target class ID
    """
    logger = get_logger()
    assert val_fold >= 0

    train_sampler, valid_sampler = None, None
    if val_ratio > 0.0: # if val_ratio is not specified then sampler is empty
        """stratified shuffle val_ratio will yield return total of n_splits,
        each val_ratio containing tuple of train and valid set with valid set
        size portion = val_ratio, while samples for each class having same
        proportions as original dataset"""

        logger.info('Validation set ratio = {}'.format(val_ratio))

        # TODO: random_state should be None so np.random is used
        # TODO: keep hardcoded n_splits=5?
        sss = StratifiedShuffleSplit(n_splits=5, test_size=val_ratio,
                                     random_state=0)
        sss = sss.split(list(range(len(trainset))), trainset.targets)

        # we have 5 plits, but will select only one of them by val_fold
        for _ in range(val_fold + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if trainset.targets[i] == target_lb]

        # NOTE: we apply random sampler for validation set as well because
        #       this set is used for training alphas for darts
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        if horovod: # train sampler for horovod
            import horovod.torch as hvd
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        logger.info('Validation set is not produced')

        # this means no sampling, validation set would be empty
        valid_sampler = SubsetSampler([])

        if horovod: # train sampler for horovod
            import horovod.torch as hvd
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    valid_sampler, num_replicas=hvd.size(), rank=hvd.rank())
        # else train_sampler is None
    return train_sampler, valid_sampler


def _add_augs(transform_train, aug:str, cutout:int):
    logger = get_logger()

    # TODO: total_aug remains None in original fastaug code
    total_aug = augs = None

    logger.info('Additional augmentation = "{}"'.format(aug))
    if isinstance(aug, list):
        transform_train.transforms.insert(0, Augmentation(aug))
    elif aug:
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
            raise ValueError('Augmentations not found: %s' % aug)

    # add cutout transform
    # TODO: use PyTorch built-in cutout
    logger.info('Cutout = {}'.format(cutout))
    if cutout > 0:
        transform_train.transforms.append(CutoutDefault(cutout))

    return total_aug, augs

def _get_imagenet_transforms():
    transform_train, transform_test = None, None

    _IMAGENET_PCA = {
        'eigval': [0.2175, 0.0188, 0.0045],
        'eigvec': [
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ]
    }

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
            interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        # TODO: Lightning is not used in original darts paper
        Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_test

