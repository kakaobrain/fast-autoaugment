from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from overrides import overrides

from .operations import Op
from .model_desc import EdgeDesc

class DagEdge(nn.Module):
    def __init__(self, desc:EdgeDesc,
                 alphas_edge:Optional['DagEdge'])->None:
        super().__init__()
        self._op = Op.create(desc.op_desc,
                             alphas_edge.alphas() if alphas_edge else [])
        self._input_ids = desc.input_ids
        self.desc = desc

    @overrides
    def forward(self, inputs:List[torch.Tensor]):
        if len(self._input_ids)==1:
            return self._op(inputs[self._input_ids[0]])
        elif len(self._input_ids) == len(inputs): # for perf
            return self._op(inputs)
        else:
            return self._op([inputs[i] for i in self._input_ids])

    def finalize(self)->Tuple[EdgeDesc, Optional[float]]:
        op_desc, rank = self._op.finalize()
        return (EdgeDesc(op_desc, self._input_ids, \
                self.desc.from_node, self.desc.to_state), rank)

    def alphas(self)->Iterable[nn.Parameter]:
        for alpha in self._op.alphas():
            if alpha is not None:
                yield alpha

    def weights(self)->Iterable[nn.Parameter]:
        for w in self._op.weights():
            yield w