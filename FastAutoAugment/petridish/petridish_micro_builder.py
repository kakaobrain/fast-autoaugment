from overrides import overrides

from  ..nas.model_desc import ModelDesc, CellDesc, CellDesc, OpDesc, \
                              EdgeDesc, ConvMacroParams, CellType
from ..nas.micro_builder import MicroBuilder
from ..nas.operations import Op
from .petridish_op import PetridishOp, PetridishFinalOp


class PetridishMicroBuilder(MicroBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('petridish_normal_op',
                    lambda op_desc, alphas: PetridishOp(op_desc, alphas, False))
        Op.register_op('petridish_reduction_op',
                    lambda op_desc, alphas: PetridishOp(op_desc, alphas, True))
        Op.register_op('petridish_final_op',
                    lambda op_desc, alphas: PetridishFinalOp(op_desc))


    @overrides
    def build(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            self._build_cell(cell_desc)

    def _build_cell(self, cell_desc:CellDesc)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # Petridish cell will start out with 1 node.
        # At each iteration we pick the last node available and add petridish op to it
        # NOTE: Where is it enforced that the cell already has 1 node. How is that node created?
        last_node_i = len(cell_desc.nodes)-1
        input_ids = list(range(last_node_i + 2)) # all previous states are input
        op_desc = OpDesc('petridish_reduction_op' if reduction else 'petridish_normal_op',
                            in_len=len(input_ids),
                            params={
                                'conv': cell_desc.conv_params,
                                # specify strides for each input, later we will
                                # give this to each primitive
                                '_strides':[2 if reduction and j < 2 else 1 \
                                           for j in input_ids],
                            })
        # add our op to last node
        node = cell_desc.nodes[last_node_i]
        edge = EdgeDesc(op_desc, index=len(node.edges),
                        input_ids=input_ids, run_mode=cell_desc.run_mode)
        node.edges.append(edge)
