from typing import Tuple, List

from overrides import overrides

from .strategy import Strategy
from .model_desc import CellDesc, CellType, NodeDesc, OpDesc, EdgeDesc


class DartsStrategy(Strategy):
    @overrides
    def _add_cell_nodes(self, cell_desc:CellDesc)->None:
        ch_out = cell_desc.get_ch_out()
        reduction = cell_desc.cell_type == CellType.Reduction

        for i in range(self.n_nodes):
            edges: List[EdgeDesc] = []
            for j in range(i+2):
                op_desc = OpDesc('mixed_op',
                                    training=self.training,
                                    ch_in=ch_out,
                                    ch_out=ch_out,
                                    stride=2 if reduction and j < 2 else 1,
                                    affine=not cell_desc.training)
                edge = EdgeDesc(op_desc,
                                input_ids=[j],
                                from_node=i,
                                to_state=j)
                edges.append(edge)
            cell_desc.nodes.append(NodeDesc(edges=edges))




