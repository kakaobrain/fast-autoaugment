from typing import Iterator, List, Sequence, Optional
from .search_cell import SearchCell, DagEdge
from .operations import MixedOp

class DartsSearchCell(SearchCell):
    def create_edge(self, ch_out:int, state_id:int, reduction:bool,
                 alphas_edge:Optional[DagEdge])->DagEdge:
        op_alphas = None if alphas_edge is None else alphas_edge.alphas
        # reduction should be used only for first 2 input node
        stride = 2 if reduction and state_id < 2 else 1
        op = MixedOp(ch_out, stride, op_alphas)

        return DagEdge(op, [state_id], op.alphas())