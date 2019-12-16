from typing import Iterator, List, Sequence, Optional

from overrides import overrides
import torch

from .search_cell import Cell, DagEdge
from .operations import Op
from .model_desc import CellDesc

class DartsCell(Cell):
    @overrides
    def create_search_edge(self, cell_desc:CellDesc, from_node:int,
            to_state:int, alphas_edge:Optional[DagEdge])->Optional[DagEdge]:
        op_alphas = None if alphas_edge is None else alphas_edge.alphas()
        # reduction should be used only for first 2 input node
        stride = 2 if reduction and state_id < 2 else 1
        op = Op.create('mixed_op', ch_in, ch_out, stride, False, op_alphas, training)
        return DagEdge(op, [state_id], op.alphas())
