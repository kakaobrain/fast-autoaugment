from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from overrides import overrides

from ..nas.model_desc import RunMode, OpDesc
from ..nas.operations import Op, FactorizedReduce


class StopForward(Op):
    def __init__(self):
        super().__init__()
        self._sg_op = StopGradient()

    @overrides
    def forward(self, x):
        y = x - self._sg_op(x)
        return y

class StopGradient(Op):
    @staticmethod
    def _zero_grad(grad):
        return torch.zeros_like(grad)

    @overrides
    def forward(self, x):
        y = x * 1
        y.register_hook(StopGradient._zero_grad)
        return y

class StopForwardReductionOp(Op):
    def __init__(self, ch_in: int, ch_out: int, affine=True):
        super().__init__()
        self._op = nn.Sequential(
            StopForward(),
            FactorizedReduce(ch_in, ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)


class StopGradientReduction(Op):
    def __init__(self, ch_in: int, ch_out: int, affine=True):
        super().__init__()
        self._op = nn.Sequential(
            StopGradient(),
            FactorizedReduce(ch_in, ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)


class PetridishOp(Op):
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

    def __init__(self, op_desc:OpDesc, alphas: Iterable[nn.Parameter], reduction:bool):
        super().__init__()

        # assume last PRIMITIVE is 'none'
        assert PetridishOp.PRIMITIVES[-1] == 'none'

        self._set_alphas(alphas, op_desc.in_len)
        self._ins = nn.ModuleList()

        for _ in range(op_desc.in_len):
            in_ops = nn.ModuleList()
            self._ins.append(in_ops)
            for primitive in PetridishOp.PRIMITIVES:
                primitive_op = Op.create(
                    OpDesc(primitive, op_desc.run_mode,
                        ch_in=op_desc.ch_in, ch_out=op_desc.ch_out,
                        stride=op_desc.stride, affine=op_desc.affine), alphas=alphas)
                op = nn.Sequential(
                    StopGradientReduction(op_desc.ch_in, op_desc.ch_out,                        affine=op_desc.affine) if reduction else StopGradient(),
                    primitive_op)
                in_ops.append(op)
        self._sf = StopForwardReductionOp(op_desc.ch_in, op_desc.ch_out,
                                          affine=op_desc.affine) if reduction else StopForward()

    @overrides
    def forward(self, x:List[Tensor]):
        s = 0
        for i, (xi, in_ops) in enumerate(zip(x, self._ins)):
            asm = F.softmax(self._alphas[i], dim=0)
            s = sum(w * op(xi) for w, op in zip(asm, in_ops)) + s
        return self._sf(s)

    @overrides
    def alphas(self) -> Iterable[nn.Parameter]:
        for alpha in self._alphas:
            yield alpha

    @overrides
    def weights(self) -> Iterable[nn.Parameter]:
        #TODO: cache this?
        for in_ops in self._ins:
            for op in in_ops:
                for w in op.parameters():
                    yield w

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        raise NotImplementedError()
        # select except 'none' op
        with torch.no_grad():
            val, i = torch.topk(self._alphas[0][:-1], 1)
        return self._ops[i].desc, float(val.item())

    @overrides
    def can_drop_path(self) -> bool:
        return False

    def _set_alphas(self, alphas: Iterable[nn.Parameter], in_len:int) -> None:
        assert len(list(self.parameters()))==0 # must call before adding other ops
        self._alphas = list(alphas)
        if not len(self._alphas):
            pl = nn.ParameterList((
                nn.Parameter(  # TODO: use better init than uniform random?
                    1.0e-3*torch.randn(len(PetridishOp.PRIMITIVES)),
                    requires_grad=True)
                for _ in range(in_len)
            ))
            self._reg_alphas = pl
            self._alphas = [p for p in self.parameters()]
