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


class MixedOp(Op):
    """The output of MixedOp is weighted output of all allowed primitives.
    """

    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'none'  # this must be at the end so top1 doesn't chose it
    ]

    def __init__(self, ch_in, ch_out, stride, affine,
                 alphas: Iterable[nn.Parameter], run_mode: RunMode):
        super().__init__()

        # assume last PRIMITIVE is 'none'
        assert MixedOp.PRIMITIVES[-1] == 'none'

        self._set_alphas(alphas)
        self._ops = nn.ModuleList()
        for primitive in MixedOp.PRIMITIVES:
            op = Op.create(
                OpDesc(primitive, run_mode, ch_in=ch_in, ch_out=ch_out,
                       stride=stride, affine=affine), alphas=alphas)
            self._ops.append(op)

    @overrides
    def forward(self, x):
        asm = F.softmax(self._alphas[0], dim=0)
        return sum(w * op(x) for w, op in zip(asm, self._ops))

    @overrides
    def alphas(self) -> Iterable[nn.Parameter]:
        for alpha in self._alphas:
            yield alpha

    @overrides
    def weights(self) -> Iterable[nn.Parameter]:
        for op in self._ops:
            for w in op.parameters():
                yield w

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        # select except 'none' op
        with torch.no_grad():
            val, i = torch.topk(self._alphas[0][:-1], 1)
        return self._ops[i].desc, float(val.item())

    @overrides
    def can_drop_path(self) -> bool:
        return False

    def _set_alphas(self, alphas: Iterable[nn.Parameter]) -> None:
        # must call before adding other ops
        assert len(list(self.parameters())) == 0
        self._alphas = list(alphas)
        if not len(self._alphas):
            new_p = nn.Parameter(  # TODO: use better init than uniform random?
                1.0e-3*torch.randn(len(MixedOp.PRIMITIVES)), requires_grad=True)
            self._reg_alphas = new_p
            self._alphas = [p for p in self.parameters()]
