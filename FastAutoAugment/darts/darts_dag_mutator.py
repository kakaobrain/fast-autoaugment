from typing import Tuple, List, Optional

from torch.utils.data import DataLoader
from overrides import overrides

from ..nas.dag_mutator import DagMutator
from ..nas.operations import Op
from ..nas.model_desc import ModelDesc, CellDesc, CellType, RunMode, OpDesc, EdgeDesc
from .mixed_op import MixedOp

class DartsDagMutator(DagMutator):
    def __init__(self) -> None:
        Op.register_op('mixed_op',
                    lambda op_desc, alphas:
                        MixedOp(op_desc.ch_in, op_desc.ch_out, op_desc.stride,
                                op_desc.affine, alphas, op_desc.run_mode)
        )

    @overrides
    def mutate(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            self._mutate_cell(cell_desc)

    def _mutate_cell(self, cell_desc:CellDesc)->None:
        ch_out = cell_desc.n_node_channels
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add mixed op for each edge
        for i, node in enumerate(cell_desc.nodes):
            for j in range(i+2):
                op_desc = OpDesc('mixed_op',
                                    run_mode=cell_desc.run_mode,
                                    ch_in=ch_out,
                                    ch_out=ch_out,
                                    stride=2 if reduction and j < 2 else 1,
                                    affine=cell_desc!=RunMode.Search)
                edge = EdgeDesc(op_desc, len(node.edges),
                                input_ids=[j],
                                from_node=i,
                                to_state=j)
                node.edges.append(edge)



