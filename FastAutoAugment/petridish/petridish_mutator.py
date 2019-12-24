from overrides import overrides

from  ..nas.model_desc import ModelDesc, CellDesc, CellDesc, OpDesc, \
                              EdgeDesc, RunMode, CellType, NodeDesc
from ..nas.dag_mutator import DagMutator
from ..nas.operations import Op
from .petridish_op import PetridishOp, PetridishFinalOp


class PetridishMutator(DagMutator):
    def __init__(self) -> None:
        Op.register_op('petridish_normal_op',
                    lambda op_desc, alphas: PetridishOp(op_desc, alphas, False))
        Op.register_op('petridish_reduction_op',
                    lambda op_desc, alphas: PetridishOp(op_desc, alphas, True))
        Op.register_op('petridish_final_op',
                    lambda op_desc, alphas: PetridishFinalOp(op_desc))


    @overrides
    def mutate(self, model_desc:ModelDesc)->None:
        for cell_desc in model_desc.cell_descs:
            self._mutate_cell(cell_desc)

    def _mutate_cell(self, cell_desc:CellDesc)->None:
        # Petridish cell will start out with 0 nodes
        # at each iteration we add one node
        ch_out = cell_desc.n_node_channels
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add mixed op for each edge
        node, i = NodeDesc(edges=[]), len(cell_desc.nodes)
        for j in range(i+2):
            op_desc = OpDesc('petridish_reduction_op' if reduction else 'petridish_normal_op',
                                run_mode=cell_desc.run_mode,
                                params={
                                    'ch_in':ch_out,
                                    'ch_out':ch_out,
                                    'stride':2 if reduction and j < 2 else 1,
                                    'affine':cell_desc!=RunMode.Search
                                })
            edge = EdgeDesc(op_desc, len(node.edges),
                            input_ids=[j],
                            from_node=i,
                            to_state=j)
            node.edges.append(edge)
        cell_desc.nodes.append(node)