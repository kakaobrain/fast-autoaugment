"""
Reference :
- https://github.com/hysts/pytorch_image_classification/blob/master/augmentations/mixup.py
- https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/imagenet_input.py#L120
"""

import numpy as np
import torch

from FastAutoAugment.metrics import CrossEntropyLabelSmooth


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1. - lam)
    assert 0.0 <= lam <= 1.0, lam
    data = data * lam + shuffled_data * (1 - lam)

    return data, targets, shuffled_targets, lam


class CrossEntropyMixUpLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(CrossEntropyMixUpLabelSmooth, self).__init__()
        self.ce = CrossEntropyLabelSmooth(num_classes, epsilon, reduction=reduction)

    def forward(self, input, target1, target2, lam):  # pylint: disable=redefined-builtin
        return lam * self.ce(input, target1) + (1 - lam) * self.ce(input, target2)
