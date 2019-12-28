from typing import Callable, Iterable, Sequence, Tuple, Dict, Optional, List
import random

from torch.utils.data import DataLoader
from overrides import overrides

from ..nas.dag_mutator import DagMutator
from ..nas.operations import Op
from ..nas.model_desc import ModelDesc, CellDesc, CellType, RunMode, OpDesc, EdgeDesc

class OpDescContainer():
    def __init__(self) -> None:
        self.normal_rand_ops:List[Tuple[OpDesc, OpDesc]] = []

    def insert_op_desc(self, op_desc_0:OpDesc, op_desc_1:OpDesc) -> None:
        self.normal_rand_ops.append((op_desc_0, op_desc_1))
        
    def get_op_desc(self) -> Iterable[Tuple[OpDesc, OpDesc]]:
        for op_desc_tuple in self.normal_rand_ops:
            yield op_desc_tuple
        
class RandomDagMutator(DagMutator):
    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'none'  # this must be at the end so top1 doesn't choose it
    ]

    def __init__(self) -> None:        
        self.op_desc_container = OpDescContainer()

    @overrides
    def mutate(self, model_desc:ModelDesc)->None:
        
        # Randomly sample a particular template that
        # will be used for all cells (normal or reduction)
        # ------------------------------------

        # Make sure we have some cells at least
        assert len(model_desc.cell_descs) > 0

        # Assuming that each cell has the same number of nodes
        # as we want each cell to be identical
        num_nodes = len(model_desc.cell_descs[-1].nodes)

        for cell_desc in model_desc.cell_descs:
            assert len(cell_desc.nodes) == num_nodes
        
        # Assuming that each cell has the same run_mode
        # TODO: Shouldn't run_mode be a property of the 
        # entire model_desc instead of each cell?
        run_mode = model_desc.cell_descs[-1].run_mode
        
        for i in range(num_nodes):
            
            # Every node connects to randomly chosen two 
            # previous states via a random operation
            connect_state_0 = random.randint(0, i+1)
            connect_state_1 = random.randint(0, i+1)

            p_ind_0 = random.randint(0, len(RandomOp.PRIMITIVES)-2)
            p_0_name = RandomDagMutator.PRIMITIVES[p_ind_0]

            p_ind_1 = random.randint(0, len(RandomOp.PRIMITIVES)-2)
            p_1_name = RandomDagMutator.PRIMITIVES[p_ind_1]
            
            params_0 = {}
            params_0['source_state'] = connect_state_0

            params_1 = {}
            params_1['source_state'] = connect_state_1

            in_len = 1
            op_desc_0 = OpDesc(p_0_name, run_mode, params_0, in_len)
            op_desc_1 = OpDesc(p_1_name, run_mode, params_1, in_len)

            self.op_desc_container.insert_op_desc(op_desc_0, op_desc_1)
    
        # Now mutate every cell according to the
        # randomly sampled template
        #------------------------------------
        for cell_desc in model_desc.cell_descs:
            self._mutate_cell(cell_desc)


    def _mutate_cell(self, cell_desc:CellDesc)->None:
        ch_out = cell_desc.n_node_channels
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add random op for each edge
        for i, node in enumerate(cell_desc.nodes):
            # Every node has two incoming edges
            op_desc_0, op_desc_1 = self.op_desc_container[i]

            op_desc_0.params['ch_in'] = ch_out
            op_desc_0.params['ch_out'] = ch_out
            op_desc_0.params['stride'] = 2 if reduction and op_desc_0.params['source_state'] < 2 else 1
            op_desc_0.params['affine'] = cell_desc!=RunMode.Search

            op_desc_1.params['ch_in'] = ch_out
            op_desc_1.params['ch_out'] = ch_out
            op_desc_1.params['stride'] = 2 if reduction and op_desc_1.params['source_state'] < 2 else 1
            op_desc_1.params['affine'] = cell_desc!=RunMode.Search

            edge_0 = EdgeDesc(op_desc_0, len(node.edges),
                                input_ids=[op_desc_0.params['source_state']],
                                from_node=-1,
                                to_state=-1)

            node.edges.append(edge_0)

            edge_1 = EdgeDesc(op_desc_1, len(node.edges),
                                input_ids=[op_desc_1.params['source_state']],
                                from_node=-1,
                                to_state=-1)

            node.edges.append(edge_1)

           


