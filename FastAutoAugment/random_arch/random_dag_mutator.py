from typing import Iterable, Sequence, Tuple, List, Set
import random

from overrides import overrides

from ..nas.dag_mutator import DagMutator
from ..nas.model_desc import ModelDesc, CellDesc, CellType, RunMode, OpDesc, EdgeDesc

class RandOps:
    """Container to store (op_names, to_states) for each nodes"""
    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        # we don't allow none edge for random ops
        # 'none'  # this must be at the end so top1 doesn't choose it
    ]

    def __init__(self, n_nodes:int, max_edges:int) -> None:
        self.ops_and_ins:List[Tuple[List[str], List[int]]] = []
        for i in range(n_nodes):
            op_names = random.choices(RandOps.PRIMITIVES, k=max_edges)
            to_states = random.sample(list(range(i+2)), k=max_edges)
            self.ops_and_ins.append((op_names, to_states))


class RandomDagMutator(DagMutator):
    @overrides
    def mutate(self, model_desc:ModelDesc)->None:
        # create random op sets for two cell types
        n_nodes = len(model_desc.cell_descs[0].nodes)
        max_edges = 2
        normal_ops, reduction_ops = RandOps(n_nodes, max_edges), RandOps(n_nodes, max_edges)

        for cell_desc in model_desc.cell_descs:
            # select rand_ops for cell type
            if cell_desc.cell_type==CellType.Regular:
                rand_ops = normal_ops
            elif cell_desc.cell_type==CellType.Reduction:
                rand_ops = reduction_ops
            else:
                raise NotImplementedError(f'CellType {cell_desc.cell_type} is not recognized')

            self._mutate_cell(cell_desc, rand_ops)

    def _mutate_cell(self, cell_desc:CellDesc, rand_ops:RandOps)->None:
        assert len(cell_desc.nodes) == len(rand_ops.ops_and_ins)

        ch_out = cell_desc.n_node_channels
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # Add random op for each edge
        for node, (op_names, to_states) in zip(cell_desc.nodes, rand_ops.ops_and_ins):
            for op_name, to_state in zip(op_names, to_states):
                op_desc = OpDesc(op_name,
                                    run_mode=cell_desc.run_mode,
                                    params={
                                        'ch_in': ch_out,
                                        'ch_out': ch_out,
                                        'stride': 2 if reduction and to_state < 2 else 1,
                                        'affine': cell_desc!=RunMode.Search
                                    })
                edge = EdgeDesc(op_desc, len(node.edges),
                                input_ids=[to_state],
                                from_node=-1,
                                to_state=-1)
                node.edges.append(edge)





