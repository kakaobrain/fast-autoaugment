from typing import Tuple, List, Optional

from torch.utils.data import DataLoader
from overrides import overrides

from ..nas.dag_mutator import DagMutator
from ..nas.operations import Op, ConvMacroParams
from ..nas.model_desc import ModelDesc, CellDesc, CellType, RunMode, OpDesc, EdgeDesc
from .mixed_op import MixedOp

class DartsDagMutator(DagMutator):
    def __init__(self) -> None:
        Op.register_op('mixed_op',
                       lambda op_desc, alphas: MixedOp(op_desc, alphas))

    @overrides
    def mutate(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            self._mutate_cell(cell_desc)

    def _mutate_cell(self, cell_desc:CellDesc)->None:
        conv_params = cell_desc.conv_params
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add mixed op for each edge
        for i, node in enumerate(cell_desc.nodes):
            for j in range(i+2):
                op_desc = OpDesc('mixed_op',
                                    params={
                                        'conv': conv_params,
                                        'stride': 2 if reduction and j < 2 else 1
                                    })
                edge = EdgeDesc(op_desc, len(node.edges),
                                input_ids=[j])
                node.edges.append(edge)



