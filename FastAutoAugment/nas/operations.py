from argparse import ArgumentError
from typing import Callable, Iterable, Tuple, Dict, Optional
from abc import ABC

from overrides import overrides, EnforceOverrides

import torch
from torch import nn, Tensor

from ..common import utils
from .model_desc import OpDesc, ConvMacroParams

# type alias
OpFactoryFn = Callable[[OpDesc, Iterable[nn.Parameter]], 'Op']

# Each op is a unary tensor operator, all take same constructor params
_ops_factory:Dict[str, OpFactoryFn] = {
    'max_pool_3x3':     lambda op_desc, alphas: PoolBN('max', op_desc),
    'avg_pool_3x3':     lambda op_desc, alphas: PoolBN('avg', op_desc),
    'skip_connect':     lambda op_desc, alphas: SkipConnect(op_desc),
    'sep_conv_3x3':     lambda op_desc, alphas: SepConv(op_desc, 3, 1),
    'sep_conv_5x5':     lambda op_desc, alphas: SepConv(op_desc, 5, 2),
    'dil_conv_3x3':     lambda op_desc, alphas: DilConv(op_desc, 3, op_desc.params['stride'], 2, 2),
    'dil_conv_5x5':     lambda op_desc, alphas: DilConv(op_desc, 5, op_desc.params['stride'], 4, 2),
    'none':             lambda op_desc, alphas: Zero(op_desc),
    'sep_conv_7x7':     lambda op_desc, alphas: SepConv(op_desc, 7, 3),
    'conv_7x1_1x7':     lambda op_desc, alphas: FacConv(op_desc, 7, 3),
    'prepr_reduce':     lambda op_desc, alphas: FactorizedReduce(op_desc),
    'prepr_normal':     lambda op_desc, alphas: ReLUConvBN(op_desc, 1, 1, 0),
    'stem_cifar':       lambda op_desc, alphas: StemCifar(op_desc),
    'stem0_imagenet':   lambda op_desc, alphas: Stem0Imagenet(op_desc),
    'stem1_imagenet':   lambda op_desc, alphas: Stem1Imagenet(op_desc),
    'pool_cifar':       lambda op_desc, alphas: PoolCifar(),
    'pool_imagenet':    lambda op_desc, alphas: PoolImagenet()
}

class Op(nn.Module, ABC, EnforceOverrides):
    @staticmethod
    def create(op_desc:OpDesc,
               alphas:Iterable[nn.Parameter]=[])->'Op':
        op = _ops_factory[op_desc.name](op_desc, alphas)
        # TODO: annotate as Final
        op.desc = op_desc # type: ignore
        return op

    @staticmethod
    def register_op(name: str, factory_fn: OpFactoryFn, exists_ok=True) -> None:
        global _ops_factory
        if name in _ops_factory:
            if not exists_ok:
                raise ArgumentError(f'{name} is already registered in op factory')
            # else no need to register again
        else:
            _ops_factory[name] = factory_fn

    # must override if op has alphas, otherwise this returns nothing!
    def alphas(self)->Iterable[nn.Parameter]:
        return # when supported, derived class should override it
        yield

    # must override if op has alphas, otherwise this will return weights + alphas!
    def weights(self)->Iterable[nn.Parameter]:
        for w in self.parameters():
            yield w

    def finalize(self)->Tuple[OpDesc, Optional[float]]:
        """for trainable op, return final op and its rank"""
        return self._desc, None

    # if op should not be dropped during drop path then return False
    def can_drop_path(self)->bool:
        return True

class PoolBN(Op):
    """AvgPool or MaxPool - BN """

    def __init__(self, pool_type:str, op_desc:OpDesc):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        affine = conv_params.affine

        stride = op_desc.params['stride']
        kernel_size = op_desc.params.get('kernel_size', 3)
        padding = op_desc.params.get('padding', 1)

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

class SkipConnect(Op):
    def __init__(self, op_desc:OpDesc) -> None:
        super().__init__()

        stride = op_desc.params['stride']
        self._op = Identity() if stride == 1 \
                              else FactorizedReduce(op_desc)

    @overrides
    def forward(self, x:Tensor)->Tensor:
        return self._op(x)

class FacConv(Op):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, op_desc:OpDesc, kernel_length:int, padding:int):
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

        stride = op_desc.params['stride']

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
    def __init__(self, op_desc:OpDesc, kernel_size:int, stride:int, padding:int):
        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

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

    def __init__(self, op_desc:OpDesc, kernel_size:int, stride:int,  padding:int, dilation:int):
        super(DilConv, self).__init__()
        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

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

    def __init__(self, op_desc:OpDesc, kernel_size:int, padding:int):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            DilConv(op_desc, kernel_size, op_desc.params['stride'], padding, dilation=1),
            DilConv(op_desc, kernel_size, 1, padding, dilation=1))

    @overrides
    def forward(self, x):
        return self.op(x)


class Identity(Op):
    def __init__(self):
        super().__init__()

    @overrides
    def forward(self, x):
        return x

    @overrides
    def can_drop_path(self)->bool:
        return False


class Zero(Op):
    """Represents no connection """

    def __init__(self, op_desc:OpDesc):
        super().__init__()
        stride = op_desc.params['stride']
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

    def __init__(self, op_desc:OpDesc):
        super(FactorizedReduce, self).__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

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
    def __init__(self, op_desc:OpDesc)->None:
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

        self._op = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to
            # BN in conv layer.
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class Stem0Imagenet(Op):
    def __init__(self, op_desc)->None:
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

        self._op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//2, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out//2, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class Stem1Imagenet(Op):
    def __init__(self, op_desc)->None:
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out
        affine = conv_params.affine

        self._op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class PoolImagenet(Op):
    def __init__(self)->None:
        super().__init__()
        self._op = nn.AvgPool2d(7)

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class PoolCifar(Op):
    def __init__(self)->None:
        super().__init__()
        self._op = nn.AdaptiveAvgPool2d(1)

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class DropPath_(nn.Module):
    """Replace values in tensor by 0. with probability p
        Ref: https://arxiv.org/abs/1605.07648
    """

    def __init__(self, p:float=0.):
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

