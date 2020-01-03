from os import stat
import torch
from torch import nn, Tensor

from overrides import overrides

from typing import Iterable, Tuple, Optional

from .cell import Cell
from .operations import Op, DropPath_
from .model_desc import ModelDesc
from ..common.common import get_logger
from ..common import utils

class Model(nn.Module):
    def __init__(self, model_desc:ModelDesc):
        super().__init__()

        logger = get_logger()

        self.desc = model_desc
        self._stem0_op = Op.create(model_desc.stem0_op)
        self._stem1_op = Op.create(model_desc.stem1_op)

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
        self.linear = nn.Linear(model_desc.cell_descs[-1].conv_params.ch_out,
                                model_desc.n_classes)

        logger.info("Total param size = %f MB", utils.param_size(self))
        logger.info(f"Alphas count = {len(set(a for a in self.alphas()))}")

    def alphas(self)->Iterable[nn.Parameter]:
        for cell in self._cells:
            if not cell.shared_alphas:
                for alpha in cell.alphas():
                    yield alpha

    def weights(self)->Iterable[nn.Parameter]:
        for cell in self._cells:
            for w in cell.weights():
                yield w

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

        # TODO: original code has slighly different way of applying stems
        s0 = self._stem0_op(x)
        s1 = self._stem1_op(x)

        logits_aux = None
        for cell in self._cells:
            #print(s0.shape, s1.shape, end='')
            s0, s1 = s1, cell.forward(s0, s1)
            #print('\t->\t', s0.shape, s1.shape)

            if cell.aux_tower is not None:
                logits_aux = cell.aux_tower(s1)

        # s1 is now the last cell's output
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1)) # flatten

        return logits, logits_aux

    def finalize(self, max_edges)->ModelDesc:
        cell_descs = [cell.finalize(max_edges=max_edges) for cell in self._cells]
        return ModelDesc(stem0_op=self.desc.stem0_op,
                         stem1_op=self.desc.stem1_op,
                         pool_op=self.desc.pool_op,
                         ds_ch=self.desc.ds_ch,
                         n_classes=self.desc.n_classes,
                         cell_descs=cell_descs)

    def drop_path_prob(self, p:float):
        """ Set drop path probability
        This will be called externally so any DropPath_ modules get
        new probability. Typically, every epoch we will reduce this probability.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p
