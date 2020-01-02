from typing import Iterable, List, Optional, Sequence, Tuple
import heapq

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
    def __init__(self, op_desc:OpDesc):
        super().__init__()
        self._op = nn.Sequential(
            StopForward(),
            FactorizedReduce(op_desc)
        )

    @overrides
    def forward(self, x):
        return self._op(x)


class StopGradientReduction(Op):
    def __init__(self, op_desc:OpDesc):
        super().__init__()
        self._op = nn.Sequential(
            StopGradient(),
            FactorizedReduce(op_desc)
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

        # assume last PRIMITIVE is 'none' (this is used for finalize)
        assert PetridishOp.PRIMITIVES[-1] == 'none'

        self._set_alphas(alphas, op_desc.in_len)
        self._edges = nn.ModuleList()

        for i in range(op_desc.in_len):
            edge = nn.ModuleList()
            self._edges.append(edge)
            op_desc.params['stride'] = op_desc.params['_strides'][i]
            for primitive in PetridishOp.PRIMITIVES:
                primitive_op = Op.create(OpDesc(primitive, op_desc.run_mode,
                                                params=op_desc.params),
                                        alphas=alphas)
                op = nn.Sequential(
                    StopGradientReduction(op_desc) if reduction else StopGradient(),
                    primitive_op)
                edge.append(op)

        self._sf = StopForwardReductionOp(op_desc) if reduction else StopForward()

    @overrides
    def forward(self, x:List[Tensor]):
        s = 0
        for i, (xi, edge) in enumerate(zip(x, self._edges)):
            edge_alphas = self._alphas[i]
            s = sum(a * op(xi) for a, op in zip(edge_alphas, edge)) + s
        return self._sf(s)

    @overrides
    def alphas(self) -> Iterable[nn.Parameter]:
        for alpha in self._alphas:
            yield alpha

    @overrides
    def weights(self) -> Iterable[nn.Parameter]:
        #TODO: cache this?
        for edge in self._edges:
            for op in edge:
                for w in op.parameters():
                    yield w

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        with torch.no_grad():
            # create list of (alpha, input_id, op_desc), sort them, select top
            # op is nn.Sequence with 2nd module as op
            l = ((a, i, op[1].desc) \
                for edge_alphas, i, edge in
                    zip(self._alphas, range(self.desc.in_len), self._edges) \
                for a, op in zip(edge_alphas, edge))
            sel = heapq.nlargest(3, l, key=lambda t: t[0])  # TODO: add config

        final_op_desc = OpDesc(name='petridish_final_op',
                                run_mode=RunMode.EvalTrain,
                                params={
                                    'ch_out': self.desc.params['ch_out'],
                                    'affine': True, # TODO: is this always right?
                                    'ins_and_ops': [(i, desc) for a, i, desc in sel]
                                },
                                in_len=self.desc.in_len
                               )

        return final_op_desc, None # rank=None to indicate no further selection

    def _set_alphas(self, alphas: Iterable[nn.Parameter], in_len:int) -> None:
        assert len(list(self.parameters()))==0 # must call before adding other ops

        self._alphas = list(alphas)

        # if this op shares alpha with another op then len should be non-zero
        if not len(self._alphas):
            pl = nn.ParameterList((
                nn.Parameter(  # TODO: use better init than uniform random?
                    torch.FloatTensor(len(PetridishOp.PRIMITIVES)).uniform_(-0.1, 0.1),
                    requires_grad=True)
                for _ in range(in_len)
            ))
            self._reg_alphas = pl # register parameters with module
            self._alphas = [p for p in self.parameters()]

class PetridishFinalOp(Op):
    def __init__(self, op_desc:OpDesc) -> None:
        super().__init__()

        ins_and_ops:Sequence[Tuple[int, OpDesc]] = op_desc.params['ins_and_ops']
        ch_out:int = op_desc.params['ch_out']
        affine:bool = op_desc.params['affine']

        self._ops = nn.ModuleList()
        self._ins:List[int] = []

        ch_out_sum = 0 #
        for i, op_desc in ins_and_ops:
            self._ops.append(Op.create(op_desc))
            ch_out_sum += op_desc.params['ch_out']
            self._ins.append(i)

        # 1x1 conv
        self._conv = nn.Conv2d(ch_out_sum, ch_out, 1,
                                stride=1, padding=0, bias=False)
        self._bn = nn.BatchNorm2d(ch_out, affine=affine)

    @overrides
    def forward(self, x:List[Tensor])->Tensor:
        res = torch.cat([op(x[i]) for op, i in zip(self._ops, self._ins)])
        res = self._conv(res)
        return self._bn(res)

