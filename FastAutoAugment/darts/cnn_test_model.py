from typing import Tuple, List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from overrides import overrides, EnforceOverrides

from .operations import create_op, get_preprocessors
from ..common.utils import first_or_default
from .dag_edge import DagEdge

class _Cell(nn.Module):
    def __init__(self, cell_desc:dict)->None:
        super().__init__()

        self.n_node_outs = cell_desc['n_node_outs']
        self.reduction = cell_desc['reduction']

        ch_pp = cell_desc['ch_pp']
        ch_p = cell_desc['ch_p']
        ch_out = cell_desc['ch_out']
        reduction_prev = cell_desc['reduction_prev']

        self._preprocess0, self._preprocess1 = \
            get_preprocessors(ch_pp, ch_p, ch_out, reduction_prev, affine=True)

        self._dag = _Cell._to_dag(cell_desc['nodes'])

    @staticmethod
    def _to_dag(nodes_desc:List[dict])->nn.ModuleList:
        dag = nn.ModuleList()
        for edges in nodes_desc:
            node = nn.ModuleList()
            for edge in edges:
                node.append(DagEdge.from_finalized(edge))
            dag.append(node)
        return dag

    @overrides
    def forward(self, s0:torch.Tensor, s1:torch.Tensor):
        s0 = self._preprocess0(s0)
        s1 = self._preprocess1(s1)

        states = [s0, s1]
        for edges in self._dag:
            # TODO: we should probably do average here otherwise output will
            #   blow up as number of primitives grows
            o = sum(edge(states) for edge in edges)
            # append one state since s is the elem-wise addition of all output
            states.append(o)

        # concat along dim=channel
        # TODO: Below assumes same shape except for channels but this won't
        #   happen for max pool etc shapes?
        # 6x[40,16,32,32]
        return torch.cat(states[-self.n_node_outs:], dim=1)
