import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs

from itertools import repeat
from functools import partial
from typing import Union, List, Tuple, Optional, Callable
import numpy as np
import math


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or weight.shape[1] != num_params):
            raise (ValueError('CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


class CondConv2d(nn.Module):
    """ Conditional Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()
        assert num_experts > 1

        if isinstance(stride, container_abcs.Iterable) and len(stride) == 1:
            stride = stride[0]
        # print('CondConv', num_experts)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = _pair(padding_val)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        num_input_fmaps = self.weight.size(1)
        num_output_fmaps = self.weight.size(0)
        receptive_field_size = 1
        if self.weight.dim() > 2:
            receptive_field_size = self.weight[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        init_weight = get_condconv_initializer(partial(nn.init.normal_, mean=0.0, std=np.sqrt(2.0 / fan_out)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            # fan_in = np.prod(self.weight_shape[1:])
            # bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(partial(nn.init.constant_, val=0), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        x_orig = x
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)     # (Expert x out x in x 3x3) --> (B x out x in x 3x3)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)                  # (B*out x in x 3 x 3)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)

        # out : (1 x B*out x ...)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])

        # out2 = self.forward_legacy(x_orig, routing_weights)
        # lt = torch.lt(torch.abs(torch.add(out, -out2)), 1e-8)
        # assert torch.all(lt), torch.abs(torch.add(out, -out2))[lt]
        # print('checked')
        return out

    def forward_legacy(self, x, routing_weights):
        # Literal port (from TF definition)
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)  # (Expert x out x in x 3x3) --> (B x out x in x 3x3)
        x = torch.split(x, 1, 0)
        weight = torch.split(weight, 1, 0)
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = torch.split(bias, 1, 0)
        else:
            bias = [None] * B
        out = []
        if self.dynamic_padding:
            conv_fn = conv2d_same
        else:
            conv_fn = F.conv2d
        for xi, wi, bi in zip(x, weight, bias):
            wi = wi.view(*self.weight_shape)
            if bi is not None:
                bi = bi.view(*self.bias_shape)
            out.append(conv_fn(
                xi, wi, bi, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups))
        out = torch.cat(out, 0)
        return out
