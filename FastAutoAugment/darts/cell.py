from FastAutoAugment.darts.operations import Op
from typing import Callable, Iterator, List, Optional, Tuple
from abc import ABC, abstractmethod

from overrides import overrides, EnforceOverrides

import torch
from torch import nn

from .dag_edge import DagEdge
from .model_desc import CellDesc, EdgeDesc, NodeDesc

class Cell(nn.Module, ABC, EnforceOverrides):
    def __init__(self, desc:CellDesc, alphas_cell:Optional['Cell']):
        super().__init__()

        self.shared_alphas = alphas_cell is not None
        self.desc = desc
        self._preprocess0 = Op.create(desc.s0_op, None)
        self._preprocess1 = Op.create(desc.s1_op, None)

        self._dag =  Cell._create_dag(desc.nodes, alphas_cell)

        self.aux_tower = None
        if desc.aux_tower_desc:
            self.aux_tower = AuxTower(desc.get_ch_out(),
                                      desc.aux_tower_desc, pool_stride=3)


    @staticmethod
    def _create_dag(nodes_desc:List[NodeDesc],
                    alphas_cell:Optional['Cell'])->nn.ModuleList:
        dag = nn.ModuleList()
        for i, node_desc in enumerate(nodes_desc):
            edges:nn.ModuleList = nn.ModuleList()
            dag.append(edges)
            for edge_desc in node_desc.edges:  # include 2 input nodes
                edges.append(DagEdge(edge_desc,
                    None if alphas_cell is None else alphas_cell._dag[i][j]))
        return dag

    def alphas(self)->Iterator[nn.Parameter]:
        return (edge.alphas                     \
            for edges in self._dag              \
                for edge in edges               \
                    if edge.alphas is not None)

    def weights(self)->Iterator[nn.Parameter]:
        return (p                               \
                for edges in self._dag          \
                    for edge in edges           \
                        for p in edge.op.parameters())

    @overrides
    def forward(self, s0, s1):
        s0 = self._preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
        s1 = self._preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]

        states = [s0, s1]
        for node in self._dag:
            # TODO: we should probably do average here otherwise output will
            #   blow up as number of primitives grows
            o = sum(edge(states) for edge in node)
            states.append(o)

        # TODO: Below assumes same shape except for channels but this won't
        #   happen for max pool etc shapes?
        return torch.cat(states[-self.desc.n_node_outs:], dim=1) # 6x[40,16,32,32]

    def finalize(self, max_edges:int)->CellDesc:
        nodes_desc:List[NodeDesc] = []
        for i, node_desc in enumerate(self.desc.nodes):
            edge_desc_ranks = [edge.finalize() for edge in self._dag[i]]
            if len(edge_desc_ranks) > max_edges:
                edge_desc_ranks = edge_desc_ranks.sort(key=lambda d:d[1],
                                             reverse=True)[:max_edges]
            nodes_desc.append(NodeDesc([edr[0] for edr in edge_desc_ranks]))

        return CellDesc(cell_type=self.desc.cell_type,
                        nodes=nodes_desc,
                        s0_op=self.desc.s0_op, s1_op=self.desc.s1_op,
                        aux_tower_desc=self.desc.aux_tower_desc,
                        n_out_nodes=self.desc.n_out_nodes,
                        n_node_channels=self.desc.n_node_channels,
                        alphas_from=self.desc.alphas_from,
                        training=False)

class AuxTower(nn.Module):
    def __init__(self, init_ch_out:int, n_classes:int, pool_stride:int):
        """assuming input size 14x14"""
        # TODO: assert input size
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=pool_stride, padding=0, count_include_pad=False),
            nn.Conv2d(init_ch_out, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # TODO: This batchnorm was omitted in orginal implementation due to a typo.
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x:torch.Tensor):
        x = self.features(x)
        x = self.linear(x.view(x.size(0), -1))
        return x
