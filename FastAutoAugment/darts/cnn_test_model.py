from typing import Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .operations import *
from ..common.utils import drop_path_
from . import genotypes as gt

class _Cell(nn.Module):
    def __init__(self, genotype:gt.Genotype, ch_pp:int, ch_p:int,
            ch_out_init:int, reduction:bool, reduction_prev:bool)->None:
        """
        We recieve genotype and build a cell that has 4 nodes, each with
        exactly two edges and only one primitive attached to this edge.
        This genotype then would be "compiled" to produce PyTorch module.
        """
        super().__init__()

        if reduction_prev: # if previous layer was reduction layer
            self._preprocess0 = FactorizedReduce(ch_pp, ch_out_init)
        else:
            self.preprocess0 = ReLUConvBN(ch_pp, ch_out_init, 1, 1, 0)
        self._preprocess1 = ReLUConvBN(ch_p, ch_out_init, 1, 1, 0)

        if reduction:
            gene, self._concat = genotype.reduce, genotype.reduce_concat
        else:
            gene, self._concat = genotype.normal, genotype.normal_concat

        self._dag = gt.to_dag(ch_out_init, gene, reduction)

    def forward(self, s0:torch.Tensor, s1:torch.Tensor, drop_prob:float):
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
        s_out = torch.cat([states[i] for i in self._concat], dim=1)

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

class CnnTestModel(nn.Module, ABC):
    def __init__(self, ch_in:int, ch_out_init:int,
            n_classes:int, n_layers:int, aux_weight:float, genotype:gt.Genotype,
            stem_multiplier=3 # 3 for Cifar, 1 for ImageNet
            ):
        super().__init__()

        self._n_layers = n_layers
        self.aux_tower, self.aux_pos = None, 2*n_layers//3
        self.genotype = genotype # will be used by derived classes

        ch_cur = stem_multiplier * ch_out_init
        self.stem0, self.stem1 = self._get_stems(ch_in, ch_cur)

        # ch_cur: output channels for cell i
        # ch_p: output channels for cell i-1
        # ch_pp: output channels for cell i-2
        ch_pp, ch_p, ch_cur = ch_cur, ch_cur, ch_out_init
        self._cells = nn.ModuleList()
        reduction_prev = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                ch_cur, reduction = ch_cur * 2, True
            else:
                reduction = False

            cell = self._get_cell(ch_pp, ch_p, ch_cur, reduction, reduction_prev)

            reduction_prev = reduction
            self._cells += [cell]
            ch_pp, ch_p = ch_p, cell.n_node_outs * ch_cur
            if aux_weight > 0. and i==self.aux_pos:
                self.aux_tower = self._get_aux_tower(ch_p, n_classes)

        self.final_pooling = self._get_final_pooling()
        self.linear = nn.Linear(ch_p, n_classes)

    def forward(self, input:torch.Tensor):
        logits_aux:torch.Tensor = None

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
    @abstractmethod
    def _get_cell(self, ch_pp:int, ch_p:int, ch_cur:int,
            reduction:bool, reduction_prev:bool)->nn.Module:
      pass

class Cifar10TestModel(CnnTestModel):
    # must have same __init__ signature as CnnTestModel
    def _get_stems(self, ch_in:int, ch_out:int)->Tuple[nn.Module, nn.Module]:
        stem = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        return stem, stem

    def _get_aux_tower(self, ch_aux:int, n_classes:int)->nn.Module:
        return AuxTower(ch_aux, n_classes, pool_stride=3)

    def _get_final_pooling(self)->nn.Module:
        return nn.AdaptiveAvgPool2d(1)

    def _get_cell(self, ch_pp:int, ch_p:int, ch_cur:int,
            reduction:bool, reduction_prev:bool)->nn.Module:
        return _Cell(self.genotype, ch_pp, ch_p, ch_cur, reduction, reduction_prev)

class ImageNetTestModel(CnnTestModel):
    # must have same __init__ signature as CnnTestModel
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

    def _get_aux_tower(self, ch_aux:int, n_classes:int)->nn.Module:
        return AuxTower(ch_aux, n_classes, pool_stride=3)

    def _get_final_pooling(self)->nn.Module:
        return nn.AvgPool2d(7)

    def _get_cell(self, ch_pp:int, ch_p:int, ch_cur:int,
            reduction:bool, reduction_prev:bool)->nn.Module:
        return _Cell(self.genotype, ch_pp, ch_p, ch_cur, reduction, reduction_prev)

