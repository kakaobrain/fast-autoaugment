from typing import Tuple, List

from overrides import overrides

from ..nas.strategy import Strategy
from ..nas.model_desc import CellDesc, CellType, NodeDesc, OpDesc, EdgeDesc


class DartsStrategy(Strategy):
    @overrides
    def _add_edges(self, cell_desc:CellDesc)->None:
        ch_out = cell_desc.n_node_channels
        reduction = (cell_desc.cell_type==CellType.Reduction)

        for i, node in enumerate(cell_desc.nodes):
            for j in range(i+2):
                op_desc = OpDesc('mixed_op',
                                    training=cell_desc.training,
                                    ch_in=ch_out,
                                    ch_out=ch_out,
                                    stride=2 if reduction and j < 2 else 1,
                                    affine=not cell_desc.training)
                edge = EdgeDesc(op_desc,
                                input_ids=[j],
                                from_node=i,
                                to_state=j)
                node.edges.append(edge)




