from typing import List, Optional

from .operations import SearchOpBase
import torch
from torch import nn

class DagEdge(nn.Module):
    def __init__(self, op:SearchOpBase, input_ids:List[int],
                 alphas:Optional[nn.Parameter])->None:
        super().__init__()
        self.op = op
        self.alphas = alphas
        self.input_ids = input_ids

    def forward(self, inputs:List[torch.Tensor]):
        if len(self.input_ids)==1:
            return self.op(inputs[self.input_ids[0]])
        elif len(self.input_ids) == len(inputs): # for perf
            return self.op(inputs)
        else:
            return self.op([inputs[i] for i in self.input_ids])

    def finalize(self)->dict:
        op_desc = self.op.finalize()
        op_desc['input_ids'] = self.input_ids
        return op_desc
