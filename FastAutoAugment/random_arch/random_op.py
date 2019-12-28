from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from overrides import overrides

from ..nas.model_desc import RunMode, OpDesc
from ..nas.operations import Op

# TODO: reduction cell might have output reduced by 2^1=2X due to
#   stride 2 through input nodes however FactorizedReduce does only
#   4X reduction. Is this correct?


class RandomOp(Op):
    """The output of RandomOp is one of the primitives chosen at uniform random
    """
    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'none'  # this must be at the end so top1 doesn't choose it
    ]

    def __init__(self, op_desc:OpDesc):
        super().__init__()

        # assume last PRIMITIVE is 'none'
        assert RandomOp.PRIMITIVES[-1] == 'none'

        self._ops = nn.ModuleList()

        # Empty alphas list since we don't know how to 
        alphas = nn.ParameterList()
        
        op = Op.create(OpDesc(RandomOp.PRIMITIVES[p_sel_ind], op_desc.run_mode, op_desc.params), alphas=None)
        self._ops.append(op)

    @overrides
    def forward(self, x):
        # There is only a single op
        return self._ops[0](x)

    @overrides
    def weights(self) -> Iterable[nn.Parameter]:
        for op in self._ops:
            for w in op.parameters():
                yield w

    
    