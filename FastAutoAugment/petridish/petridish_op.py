from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Tuple
import heapq

import torch
from torch import Tensor, nn

from overrides import overrides

from ..nas.model_desc import ConvMacroParams, OpDesc
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

        # create alphas for the op
        self._set_alphas(alphas, op_desc.in_len)

        # create edges for the op, each edge connects input state,
        # within each edge we will have all N primitives
        self._edges = nn.ModuleList()

        for i in range(op_desc.in_len):
            # edge contains all primitives with alphas
            edge = nn.ModuleList()
            self._edges.append(edge)

            # for each input stride could be different,
            # so we will make copy of our params and then set stride for this input
            params = deepcopy(op_desc.params)
            params['stride'] = op_desc.params['_strides'][i]

            # create primitives for the edge
            for primitive in PetridishOp.PRIMITIVES:
                primitive_op = Op.create(OpDesc(primitive, params=params),
                                        alphas=alphas)
                # wrap primitive with sg
                op = nn.Sequential(StopGradient(), primitive_op)
                edge.append(op)

        # TODO: check with Dey: Do we really need StopForwardReductionOp
        #   or StopGradientReductionOp because these two will only make sense
        #   for cell stems.
        self._sf = StopForward()

    @overrides
    def forward(self, x:List[Tensor]):
        s = 0.0
        # apply each input in the list to associated edge
        for i, (xi, edge) in enumerate(zip(x, self._edges)):
            # apply input to each primitive within edge
            # TODO: is avg better idea than sum here? sum can explode as
            #   number of primitives goes up
            s = sum(a * op(xi) for a, op in zip(self._alphas[i], edge)) + s
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
        with torch.no_grad(): # probably this is not needed
            # Create list of (alpha, input_id, op_desc), sort them, select top k.
            # Here op should be nn.Sequence of sg followed by primitive.
            # First for loop gets edge and associated alphas.
            # Second for loop gets op and associated alpha.
            l = ((a, i, op[1].desc)                                         \
                for edge_alphas, i, edge in                                 \
                    zip(self._alphas, range(self.desc.in_len), self._edges) \
                for a, op in zip(edge_alphas, edge))

            # select 3 largest ops by alpha
            sel = heapq.nlargest(3, l, key=lambda t: t[0])  # TODO: add config

        # PetridishFinalOp needs to know each input and associated primitive
        final_op_desc = OpDesc(name='petridish_final_op',
                                params={
                                    'conv': self.desc.params['conv'],
                                    'ins_and_ops': [(i, desc) for a, i, desc in sel]
                                },
                                # Number of inputs remains same although only 3 of
                                # them will be used.
                                in_len=self.desc.in_len
                               )

        # rank=None to indicate no further selection needed as in darts
        return final_op_desc, None

    def _set_alphas(self, alphas: Iterable[nn.Parameter], in_len:int) -> None:
        assert len(list(self.parameters()))==0 # must call before adding other ops

        # If we are using shared alphas from another cell, don't create our own
        self._alphas = list(alphas)
        if not len(self._alphas):
            # Each nn.Parameter is tensor with alphas for entire edge.
            # We will create same numbers of nn.Parameter as number of edges
            pl = nn.ParameterList((
                nn.Parameter(  # TODO: use better init than uniform random?
                    torch.FloatTensor(len(PetridishOp.PRIMITIVES)).uniform_(-0.1, 0.1),
                    requires_grad=True)
                for _ in range(in_len)
            ))
            # register parameters with module
            self._reg_alphas = pl
            # save PyTorch registered alphas into list for later use
            self._alphas = [p for p in self.parameters()]

class PetridishFinalOp(Op):
    def __init__(self, op_desc:OpDesc) -> None:
        super().__init__()

        # get list of inputs and associated primitives
        ins_and_ops:Sequence[Tuple[int, OpDesc]] = op_desc.params['ins_and_ops']
        # conv params typically specified by macro builder
        conv_params:ConvMacroParams = op_desc.params['conv']

        self._ops = nn.ModuleList()
        self._ins:List[int] = []

        for i, op_desc in ins_and_ops:
            op_desc.params['conv'] = conv_params
            self._ops.append(Op.create(op_desc))
            self._ins.append(i)

        # number of channels as we will concate output of ops
        ch_out_sum = conv_params.ch_out * len(self._ins)

        # Apply 1x1 conv to reduce back channels to as specified by macro builder
        self._conv = nn.Conv2d(ch_out_sum, conv_params.ch_out, 1,
                                stride=1, padding=0, bias=False)
        self._bn = nn.BatchNorm2d(conv_params.ch_out, affine=conv_params.affine)

    @overrides
    def forward(self, x:List[Tensor])->Tensor:
        res = torch.cat([op(x[i]) for op, i in zip(self._ops, self._ins)], dim=1)
        res = self._conv(res)
        return self._bn(res)

