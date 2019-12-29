from typing import Callable, Iterable, Sequence, Tuple, Dict, Optional, List, Set
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

    def _sample_without_replacement(self, low:int, high:int, num_to_sample:int) -> Set[int]:
        assert high - low + 1 >= num_to_sample

        connect_ids = set()
        while len(connect_ids) < num_to_sample:
            state_id = random.randint(low, high)
            connect_ids.add(state_id)

        return connect_ids

    def _sample_ops(self, num_to_sample:int) -> List[str]:

        ops_selected = []
        for i in range(num_to_sample):
            p_ind = random.randint(0, len(RandomDagMutator.PRIMITIVES)-2)
            p_name = RandomDagMutator.PRIMITIVES[p_ind]
            ops_selected.append(p_name)

        return ops_selected
        

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

        # Number of edges going in to a state
        # TODO: Make into a parameter
        num_incoming = 2
        
        for i in range(num_nodes):
            
            connect_ids = self._sample_without_replacement(0, i+1, num_incoming)

            op_names = self._sample_ops(num_incoming)

            in_len = 1
            op_descs = []
            for id_and_op in zip(connect_ids, op_names):
                params = {}
                params['source_state'] = id_and_op[0]
                op_desc = OpDesc(id_and_op[1], run_mode, params, in_len)
                op_descs.append(op_desc)

            self.op_desc_container.insert_op_desc(tuple(op_descs))
            
        # Now mutate every cell according to the
        # random template
        #------------------------------------
        for cell_desc in model_desc.cell_descs:
            self._mutate_cell(cell_desc)


    def _mutate_cell(self, cell_desc:CellDesc)->None:
        ch_out = cell_desc.n_node_channels
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # Add random op for each edge
        for i, node in enumerate(cell_desc.nodes):
            # Every node has K incoming edges
            op_descs = self.op_desc_container[i]
        
            for op_desc in op_descs:
                op_desc.params['ch_in'] = ch_out
                op_desc.params['ch_out'] = ch_out
                op_desc.params['stride'] = 2 if reduction and op_desc.params['source_state'] < 2 else 1
                op_desc.params['affine'] = cell_desc!=RunMode.Search    

                edge = EdgeDesc(op_desc, len(node.edges),
                                input_ids=[op_desc.params['source_state']],
                                from_node=-1,
                                to_state=-1)

                node.edges.append(edge)


           


