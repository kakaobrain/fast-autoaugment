from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import genotypes
from ..common import utils


# Each op is a uninary tensor operator, all take same constructor params
_ops_factory = {
    'max_pool_3x3': lambda ch, stride, affine, alphas, training: PoolBN('max', ch, 3, stride, 1, affine=affine),
    'avg_pool_3x3': lambda ch, stride, affine, alphas, training: PoolBN('avg', ch, 3, stride, 1, affine=affine),
    'skip_connect': lambda ch, stride, affine, alphas, training: Identity(training) if stride == 1 else FactorizedReduce(ch, ch, affine=affine),
    'sep_conv_3x3': lambda ch, stride, affine, alphas, training: SepConv(ch, ch, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda ch, stride, affine, alphas, training: SepConv(ch, ch, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda ch, stride, affine, alphas, training: DilConv(ch, ch, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda ch, stride, affine, alphas, training: DilConv(ch, ch, 5, stride, 4, 2, affine=affine),
    'none':         lambda ch, stride, affine, alphas, training: Zero(stride),
    'sep_conv_7x7': lambda ch, stride, affine, alphas, training: SepConv(ch, ch, 7, stride, 3, affine=affine),
    'conv_7x1_1x7': lambda ch, stride, affine, alphas, training: FacConv(ch, ch, 7, stride, 3, affine=affine),
    'mixed_op':     lambda ch, stride, affine, alphas, training: MixedOp(ch, stride, affine, alphas)
}

class SearchOpBase(nn.Module, ABC):
    def alphas(self)->Optional[nn.Parameter]:
        return None

    def finalize(self)->dict:
        return self._create_info

    def _set_create_info(self, create_info:dict)->None:
        self._create_info = create_info


def create_op(name:str, ch:int, stride:int, affine:bool,
               alphas: Optional[nn.Parameter]=None, training=True)->SearchOpBase:
    op = _ops_factory[name](ch, stride, affine, alphas, training)
    op._set_create_info({'name':name, 'ch':ch, 'stride':stride, 'affine':affine })
    return op


class PoolBN(SearchOpBase):
    """AvgPool or MaxPool - BN """

    def __init__(self, pool_type, ch, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(ch, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class FacConv(SearchOpBase):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, ch_in, ch_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, (kernel_length, 1),
                      stride, padding, bias=False),
            nn.Conv2d(ch_in, ch_out, (1, kernel_length),
                      stride, padding, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class ReLUConvBN(SearchOpBase):
    """
    Stack of relu-conv-bn
    """

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, affine=True):
        """

        :param ch_in:
        :param ch_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param affine:
        """
        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(SearchOpBase):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,
                      groups=ch_in, bias=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(SearchOpBase):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2

    This is same as two DilConv stacked with dilation=1
    """

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=ch_in, bias=False),
            nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_in, affine=affine),

            # repeat above but with stride 1
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=ch_in, bias=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(SearchOpBase):
    def __init__(self, training):
        super().__init__()
        self._drop_op = DropPath_() if not training else None

    def forward(self, x):
        # TODO: investigate need for drop path
        if self._drop_op is not None:
            x = self._drop_op(x)
        return x


class Zero(SearchOpBase):
    """Represents no connection """

    def __init__(self, stride):
        super().__init__()

        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(SearchOpBase):
    """
    reduce feature maps height/width by 4X while keeping channel same using two 1x1 convs, each with stride=2.
    """
    # TODO: modify to take number of nodes in reduction cells where stride 2 was applied (currently only first two input nodes)

    def __init__(self, ch_in, ch_out, affine=True):
        super(FactorizedReduce, self).__init__()

        assert ch_out % 2 == 0

        self.relu = nn.ReLU()
        # this conv layer operates on even pixels to produce half width, half channels
        self.conv_1 = nn.Conv2d(ch_in, ch_out // 2, 1,
                                stride=2, padding=0, bias=False)
        # this conv layer operates on odd pixels (because of code in forward()) to produce half width, half channels
        self.conv_2 = nn.Conv2d(ch_in, ch_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)

        # x: torch.Size([32, 32, 32, 32])
        # conv1: [b, c_out//2, d//2, d//2]
        # conv2: []
        # out: torch.Size([32, 32, 16, 16])

        # concate two half channels to produce same number of channels as before but with output as only half the width
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(SearchOpBase):
    """
    The output of MixedOp is weighted output of all allowed primitives.
    """

    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect', #identity
        'sep_conv_3x3',
        'sep_conv_5x5',
            'dil_conv_3x3',
        'dil_conv_5x5',
        'none' # this must be at the end so top1 doesn't chose it
    ]


    def __init__(self, ch, stride, affine, alphas: Optional[nn.Parameter]):
        super().__init__()

        assert MixedOp.PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'
        self.ch, self.stride = ch, stride
        self._ops = nn.ModuleList()
        if alphas is None:
            alphas = nn.Parameter( # TODO: use better init than uniform random?
                1e-3*torch.randn(len(MixedOp.PRIMITIVES)), requires_grad=True)
        self._alphas = alphas

        for primitive in MixedOp.PRIMITIVES:
            # create corresponding layer
            op = create_op(primitive, ch, stride, affine)
            self._ops.append(op)

    def forward(self, x):
        asm = F.softmax(self._alphas)
        return sum(w * op(x) for w, op in zip(asm, self._ops))

    # overrides
    def alphas(self)->Optional[nn.Parameter]:
        return self._alphas

    # overrides
    def finalize(self)->dict:
        # select except 'none' op
        val, i = torch.topk(self._alphas[:-1], 1)
        op_desc = self._ops[i].finalize()
        op_desc['rank'] = val
        return op_desc


class DropPath_(nn.Module):
    """Replace values in tensor by 0. with probability p"""

    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        return utils.drop_path_(x, self.p, self.training)
