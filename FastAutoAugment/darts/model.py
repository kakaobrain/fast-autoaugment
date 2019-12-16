import torch
from torch import nn, Tensor
import torch.nn.functional as F
import logging
from typing import List, Optional, Iterator, Tuple

from overrides import overrides

from .cell import Cell
from .operations import Op, DropPath_
from .model_desc import ModelDesc

class Model(nn.Module):
    def __init__(self, model_desc:ModelDesc):
        super().__init__()

        self.desc = model_desc
        self._stem0_op = Op.create(model_desc.stem0_op, None)
        self._stem1_op = Op.create(model_desc.stem1_op, None)

        self._cells = nn.ModuleList()

        for i, cell_desc in enumerate(model_desc.cell_descs):
            alphas_cell = None if cell_desc.alphas_from==i  \
                               else self._cells[cell_desc.alphas_from]
            cell = Cell(cell_desc, alphas_cell)
            self._cells.append(cell)

        # adaptive pooling output size to 1x1
        self.final_pooling = Op.create(model_desc.pool_op)
        # since ch_p records last cell's output channels
        # it indicates the input channel number
        self.linear = nn.Linear(model_desc.cell_descs[-1].get_ch_out(),
                                model_desc.n_classes)

    def alphas(self)->Iterator[nn.Parameter]:
        return (alpha                               \
                    for cell in self._cells         \
                        if not cell.shared_alphas   \
                            for alpha in cell.alphas())
    def weights(self)->Iterator[nn.Parameter]:
        return (p                                   \
                for cell in self._cells             \
                    for p in cell.weights())

    @overrides
    def forward(self, x)->Tuple[Tensor, Optional[Tensor]]:
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

        # TODO: original code has s0==s1 for cifar but not for imagenet
        s0 = self._stem0_op(input)
        s1 = self._stem1_op(s0)

        logits_aux = None
        for cell in self._cells:
            s0, s1 = s1, cell.forward(s0, s1)
            if cell.aux_tower is not None:
                logits_aux = cell.aux_tower(s1)

        # s1 is now the last cell's output
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1)) # flatten

        return logits, logits_aux

    def finalize(self, max_edges)->ModelDesc:
        cell_descs = [cell.finalize() for cell in self._cells]
        return ModelDesc(stem0_op=self.desc.stem0_op,
                         stem1_op=self.desc.stem1_op,
                         pool_op=self.desc.pool_op,
                         ch_in=self.desc.ch_in,
                         n_classes=self.desc.n_classes,
                         cell_descs=cell_descs)

    def drop_path_prob(self, p):
        """ Set drop path probability
        This will be called exteranlly so any DropPath_ modules get
        new probability. Typically, every epoch we will reduce this probability.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

