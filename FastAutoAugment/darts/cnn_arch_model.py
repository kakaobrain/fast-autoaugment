import torch
from torch import nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Iterator

from .search_cell import SearchCell
from . import genotypes as gt

class _CnnModel(nn.Module):
    """ Search CNN model """

    def __init__(self, ch_in:int, ch_out_init:int, n_classes:int, n_layers:int,
        n_nodes=4, n_node_outs=4, stem_multiplier=3):
        """

        :param ch_in: number of channels in input image (3)
        :param ch_out_init: number of output channels from the first layer) /
            stem_multiplier (16)
        :param n_classes: number of classes
        :param n_layers: number of cells of current network
        :param n_nodes: nodes inside cell
        :param n_node_outs: output channel of cell = n_node_outs * ch
        :param stem_multiplier: output channel of stem net = stem_multiplier*ch
        """
        super().__init__()

        self.ch_in = ch_in
        self.ch_out_init = ch_out_init
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_node_outs = n_node_outs

        # stem is the start of network. This is additional
        # 3x3 conv layer that multiplies channels
        # TODO: why do we need stem_multiplier?
        ch_cur = stem_multiplier * ch_out_init # 3*16
        self._stem = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to
            # BN in conv layer.
            nn.Conv2d(ch_in, ch_cur, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_cur)
        )

        # ch_cur: output channels for cell i
        # ch_p: output channels for cell i-1
        # ch_pp: output channels for cell i-2
        ch_pp, ch_p, ch_cur = ch_cur, ch_cur, ch_out_init # 48, 48, 16
        self._cells = nn.ModuleList()
        first_normal:Optional[SearchCell] = None
        first_reduction:Optional[SearchCell] = None
        reduction_prev = False
        for i in range(n_layers):
            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [n_layers // 3, 2 * n_layers // 3]:
                ch_cur, reduction = ch_cur * 2, True
            else:
                reduction = False

            # [ch_p, h, h] => [n_node_outs*ch_cur, h/h//2, h/h//2]
            # the output channels = n_node_outs * ch_cur
            cell = SearchCell(n_nodes, n_node_outs, ch_pp, ch_p, ch_cur, reduction,
                reduction_prev, first_reduction if reduction else first_normal)
            # update reduction_prev
            reduction_prev = reduction

            self._cells.append(cell)
            if first_normal is None and not reduction:
                first_normal = cell
            if first_reduction is None and reduction:
                first_reduction = cell
            ch_pp, ch_p = ch_p, n_node_outs * ch_cur

        # adaptive pooling output size to 1x1
        self.final_pooling = nn.AdaptiveAvgPool2d(1)
        # since ch_p records last cell's output channels
        # it indicates the input channel number
        self.linear = nn.Linear(ch_p, n_classes)

    def get_alphas(self)->Iterator[nn.Parameter]:
        return self._cells[0].get_alphas() # all other cells shares alphas

    def forward(self, x):
        """
        Runs x through cells with alphas, applies final pooling, send through
            FCs and returns logits.

        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        """

        # first two inputs
        s0 = s1 = self._stem(x) # [b, 3, 32, 32] => [b, 48, 32, 32]
        # macro structure: each cell consumes output of
        # previous two cells
        for cell in self._cells:
            s0, s1 = s1, cell.forward(s0, s1) # [40, 64, 32, 32]

        # s1 is now the last cell's output
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1)) # flatten

        return logits

class CnnArchModel(nn.Module):
    def __init__(self, ch_in, ch_out_init, n_classes, n_layers,
            n_nodes=4, n_node_outs=4, stem_multiplier=3):
        super().__init__()
        self.n_nodes = n_nodes

        self._model = _CnnModel(ch_in, ch_out_init, n_classes, n_layers,
            n_nodes, n_node_outs, stem_multiplier)

    def alphas(self) -> Iterator[nn.Parameter]:
        return self._model.get_alphas()

    def forward(self, x):
        return self._model(x)

    def genotype(self):
        gene_normal = gt.parse(self._alphas_normal, k=2)
        gene_reduce = gt.parse(self._alphas_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self._model.parameters()


