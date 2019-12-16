from typing import Callable, List, Optional, Tuple, Dict, Optional, Final
from abc import ABC, abstractmethod
import copy

from overrides import overrides, EnforceOverrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import utils
from .model_desc import OpDesc

# Each op is a uninary tensor operator, all take same constructor params
_ops_factory:Dict[str, Callable[[OpDesc, Optional[nn.Parameter]], 'Op']] = {
    'max_pool_3x3':     lambda op_desc, alphas:
                        PoolBN('max', op_desc.ch_in, 3, op_desc.stride, 1, affine=op_desc.affine),
    'avg_pool_3x3':     lambda op_desc, alphas:
                        PoolBN('avg', op_desc.ch_in, 3, op_desc.stride, 1, affine=op_desc.affine),
    'skip_connect':     lambda op_desc, alphas:
                        Identity(op_desc.training) if op_desc.stride == 1 else \
                            FactorizedReduce(op_desc.ch_in, op_desc.ch_out, affine=op_desc.affine),
    'sep_conv_3x3':     lambda op_desc, alphas:
                        SepConv(op_desc.ch_in, op_desc.ch_out, 3, op_desc.stride, 1, affine=op_desc.affine),
    'sep_conv_5x5':     lambda op_desc, alphas:
                        SepConv(op_desc.ch_in, op_desc.ch_out, 5, op_desc.stride, 2, affine=op_desc.affine),
    'dil_conv_3x3':     lambda op_desc, alphas:
                        DilConv(op_desc.ch_in, op_desc.ch_out, 3, op_desc.stride, 2, 2, affine=op_desc.affine),
    'dil_conv_5x5':     lambda op_desc, alphas:
                        DilConv(op_desc.ch_in, op_desc.ch_out, 5, op_desc.stride, 4, 2, affine=op_desc.affine),
    'none':             lambda op_desc, alphas:
                        Zero(op_desc.stride),
    'sep_conv_7x7':     lambda op_desc, alphas:
                        SepConv(op_desc.ch_in, op_desc.ch_out, 7, op_desc.stride, 3, affine=op_desc.affine),
    'conv_7x1_1x7':     lambda op_desc, alphas:
                        FacConv(op_desc.ch_in, op_desc.ch_out, 7, op_desc.stride, 3, affine=op_desc.affine),
    'mixed_op':         lambda op_desc, alphas:
                        MixedOp(op_desc.ch_in, op_desc.ch_out, op_desc.stride,
                                op_desc.affine, alphas, op_desc.training),
    'prepr_reduce':     lambda op_desc, alphas:
                        FactorizedReduce(op_desc.ch_in, op_desc.ch_out,
                                affine=op_desc.affine or not op_desc.training),
    'prepr_normal':     lambda op_desc, alphas:
                        ReLUConvBN(op_desc.ch_in, op_desc.ch_out, 1, 1, 0,
                                affine=op_desc.affine or not op_desc.training),
    'stem_cifar':       lambda op_desc, alphas:
                        StemCifar(op_desc.ch_in, op_desc.ch_out, affine=op_desc.affine),
    'stem0_imagenet':   lambda op_desc, alphas:
                        Stem0Imagenet(op_desc.ch_in, op_desc.ch_out, affine=op_desc.affine),
    'stem1_imagenet':   lambda op_desc, alphas:
                        Stem1Imagenet(op_desc.ch_in, op_desc.ch_out, affine=op_desc.affine),
    'pool_cifar':       lambda op_desc, alphas:
                        PoolCifar(),
    'pool_imagenet':    lambda op_desc, alphas:
                        PoolImagenet()

}

class Op(nn.Module, ABC, EnforceOverrides):
    def alphas(self)->Optional[nn.Parameter]:
        return None

    @staticmethod
    def create(op_desc:OpDesc, alphas: Optional[nn.Parameter]=None)->'Op':
        op = _ops_factory[op_desc.name](op_desc, alphas)
        op.desc = op_desc # TODO: annotate as Final
        return op

    def alphas(self)->Optional[nn.Parameter]:
        return None # when supported, derived class should override it

    def finalize(self)->Tuple[OpDesc, Optional[float]]:
        """for trainable op, return final op and its rank"""
        return self._desc, None

class PoolBN(Op):
    """AvgPool or MaxPool - BN """

    def __init__(self, pool_type, ch_in, kernel_size, stride, padding, affine=True):
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

        self.bn = nn.BatchNorm2d(ch_in, affine=affine)

    @overrides
    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out

class FacConv(Op):
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

    @overrides
    def forward(self, x):
        return self.net(x)


class ReLUConvBN(Op):
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

    @overrides
    def forward(self, x):
        return self.op(x)


class DilConv(Op):
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

    @overrides
    def forward(self, x):
        return self.op(x)


class SepConv(Op):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2

    This is same as two DilConv stacked with dilation=1
    """

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            DilConv(ch_in, ch_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(ch_in, ch_out, kernel_size, 1, padding, dilation=1, affine=affine))

    @overrides
    def forward(self, x):
        return self.op(x)


class Identity(Op):
    def __init__(self, training:bool):
        super().__init__()
        self._drop_op = DropPath_() if not training else None

    @overrides
    def forward(self, x):
        # TODO: investigate need for drop path
        if self._drop_op is not None:
            x = self._drop_op(x)
        return x


class Zero(Op):
    """Represents no connection """

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    @overrides
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(Op):
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

    @overrides
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

class StemCifar(Op):
    def __init__(self, ch_in, ch_out, affine)->None:
        super().__init__()
        self._op = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to
            # BN in conv layer.
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

class Stem0Imagenet(Op):
    def __init__(self, ch_in, ch_out, affine)->None:
        super().__init__()
        self._op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out//2, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

class Stem1Imagenet(Op):
    def __init__(self, ch_in, ch_out, affine)->None:
        super().__init__()
        self._op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

class PoolImagenet(Op):
    def __init__(self)->None:
        super().__init__()
        self._op = nn.AvgPool2d(7)

    @overrides
    def forward(self, x):
        return self._op(x)

class PoolCifar(Op):
    def __init__(self)->None:
        super().__init__()
        self._op = nn.AdaptiveAvgPool2d(1)

    @overrides
    def forward(self, x):
        return self._op(x)

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

    @overrides
    def forward(self, x):
        return utils.drop_path_(x, self.p, self.training)

# TODO: reduction cell might have output reduced by 2^1=2X due to
#   stride 2 through input nodes however FactorizedReduce does only
#   4X reduction. Is this correct?


class MixedOp(Op):
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


    def __init__(self, ch_in, ch_out, stride, affine,
                 alphas:Optional[nn.Parameter], training:bool):
        super().__init__()

        assert MixedOp.PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

        self._ops = nn.ModuleList()
        if alphas is None:
            alphas = nn.Parameter( # TODO: use better init than uniform random?
                1.0e-3*torch.randn(len(MixedOp.PRIMITIVES)), requires_grad=True)
        self._alphas = alphas

        for primitive in MixedOp.PRIMITIVES:
            # create corresponding layer
            op = Op.from_desc(
                OpDesc(primitive, training, ch_in, ch_out, stride, affine), alphas)
            self._ops.append(op)

    @overrides
    def forward(self, x):
        asm = F.softmax(self._alphas)
        return sum(w * op(x) for w, op in zip(asm, self._ops))

    @overrides
    def alphas(self)->Optional[nn.Parameter]:
        return self._alphas

    @overrides
    def finalize(self)->Tuple[OpDesc, Optional[float]]:
        # select except 'none' op
        with torch.no_grad():
            val, i = torch.topk(self._alphas[:-1], 1)
        return self._ops[i].desc, float(val.item())
