from typing import Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from overrides import overrides, EnforceOverrides

from .operations import create_op, FactorizedReduce, ReLUConvBN
from ..common.utils import first_or_default
from .dag_edge import DagEdge

class _Cell(nn.Module):
    def __init__(self, cell_desc:dict)->None:
        super().__init__()

        self.n_node_outs = cell_desc['n_node_outs']
        self.ch_pp = cell_desc['ch_pp']
        self.ch_p = cell_desc['ch_p']
        self.ch_out_init = cell_desc['ch_out_init']
        self.reduction = cell_desc['reduction']
        self.reduction_prev = cell_desc['reduction_prev']
        self.ch_out = cell_desc['ch_out']

        if self.reduction_prev: # if previous layer was reduction layer
            self._preprocess0 = FactorizedReduce(self.ch_pp,self. ch_out_init)
        else:
            self._preprocess0 = ReLUConvBN(self.ch_pp, self.ch_out_init, 1, 1, 0)
        self._preprocess1 = ReLUConvBN(self.ch_p, self.ch_out_init, 1, 1, 0)

        self._dag = self._to_dag(self.ch_out_init, cell_desc, self.reduction)

    def _to_dag(self, ch_in:int, cell_desc:dict, reduction:bool)->nn.ModuleList:
        """ generate discrete ops from gene """
        dag = nn.ModuleList()
        for edges in cell_desc['nodes']:
            row = nn.ModuleList()
            for edge in edges:
                op = create_op(edge['name'], edge['ch'], edge['stride'], edge['affine'])
                row.append(DagEdge(op, edge['input_ids'], None))
                op_name, s_idx = , edge['input_ids']
                # reduction cell & from input nodes => stride = 2
                stride = 2 if reduction and s_idx < 2 else 1
            dag.append(row)

        return dag

    @overrides
    def forward(self, s0:torch.Tensor, s1:torch.Tensor):
        s0 = self._preprocess0(s0)
        s1 = self._preprocess1(s1)

        states = [s0, s1]
        for edges in self._dag:
            # for each noce i, find which previous two node we
            # connect to and corresponding ops for them
            # aggregation of ops result is arithmatic sum
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        # concatenate outputs of all node which becomes the result of the cell
        # this makes it necessory that wxh is same for all outputs
        node_outs_range = range(len(states))[:self.n_nodes_out]
        node_outs = [states[i] for i in node_outs_range]
        s_out = torch.cat(node_outs, dim=1)

        return s_out

class AuxTower(nn.Module):
    def __init__(self, ch_out_init:int, n_classes:int, pool_stride:int):
        """assuming input size 14x14"""
        # TODO: assert input size
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=pool_stride, padding=0, count_include_pad=False),
            nn.Conv2d(ch_out_init, 128, 1, bias=False),
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

class CnnTestModel(nn.Module, ABC, EnforceOverrides):
    def __init__(self, ch_in:int, ch_out_init:int,
            n_classes:int, n_layers:int, aux_weight:float, model_desc:dict,
            stem_multiplier=3 # 3 for Cifar, 1 for ImageNet
            ):
        super().__init__()

        self._n_layers = n_layers
        self.aux_tower, self.aux_pos = None, 2*n_layers//3
        self.model_desc = model_desc # will be used by derived classes

        ch_cur = stem_multiplier * ch_out_init
        self.stem0, self.stem1 = self._get_stems(ch_in, ch_cur)

        # ch_cur: output channels for cell i
        # ch_p: output channels for cell i-1
        # ch_pp: output channels for cell i-2
        self._cells = nn.ModuleList()
        for i, cell_desc in enumerate(model_desc['cells']):
            cell = _Cell(cell_desc)
            self._cells += [cell]
            if aux_weight > 0. and i==self.aux_pos:
                self.aux_tower = self._get_aux_tower(
                    cell.n_node_outs * cell.ch_out, n_classes)

        self.final_pooling = self._get_final_pooling()
        self.linear = nn.Linear(cell.n_node_outs * cell.ch_out, n_classes)

    @overrides
    def forward(self, input:torch.Tensor):
        logits_aux = None

        # TODO: this is bit weired logic in original paper
        s0 = self.stem0(input)
        if self.stem0 == self.stem1:
            s1 = s0
        else:
            s1 = self.stem1(s0)

        for i, cell in enumerate(self._cells):
            s0, s1 = s1, cell(s0, s1)
            if self.training and self.aux_tower and i==self.aux_pos:
                logits_aux = self.aux_tower(s1)
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1))
        return logits, logits_aux

    def drop_path_prob(self, p):
        """ Set drop path probability
        This will be called exteranlly so any DropPath_ modules get
        new probability. Typically, every epoch we will reduce this probability.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

    # Abstract methods
    @abstractmethod
    def _get_stems(self, ch_in:int, ch_out:int)->Tuple[nn.Module, nn.Module]:
        pass
    @abstractmethod
    def _get_aux_tower(self, ch_aux:int, n_classes:int)->nn.Module:
        pass
    @abstractmethod
    def _get_final_pooling(self)->nn.Module:
      pass

class Cifar10TestModel(CnnTestModel):
    # must have same __init__ signature as CnnTestModel
    @overrides
    def _get_stems(self, ch_in:int, ch_out:int)->Tuple[nn.Module, nn.Module]:
        stem = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        return stem, stem

    @overrides
    def _get_aux_tower(self, ch_aux:int, n_classes:int)->nn.Module:
        return AuxTower(ch_aux, n_classes, pool_stride=3)

    @overrides
    def _get_final_pooling(self)->nn.Module:
        return nn.AdaptiveAvgPool2d(1)

class ImageNetTestModel(CnnTestModel):
    # must have same __init__ signature as CnnTestModel
    @overrides
    def _get_stems(self, ch_in:int, ch_out:int)->Tuple[nn.Module, nn.Module]:
        stem0 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out//2, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        return stem0, stem1

    @overrides
    def _get_aux_tower(self, ch_aux:int, n_classes:int)->nn.Module:
        return AuxTower(ch_aux, n_classes, pool_stride=3)

    @overrides
    def _get_final_pooling(self)->nn.Module:
        return nn.AvgPool2d(7)
