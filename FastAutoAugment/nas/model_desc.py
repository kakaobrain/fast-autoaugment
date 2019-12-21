from enum import Enum
from typing import Optional, List
import yaml

class DescBase:
    def serialize(self)->str:
        return yaml.dump(self)
    def deserialize(self, v:str)->'DescBase':
        return yaml.load(v, Loader=yaml.Loader)


class RunMode(Enum):
    Search = 'search'
    EvalTrain = 'eval_train'
    EvalTest = 'eval_test'

class OpDesc(DescBase):
    """Op description that is in each edge
    """
    def __init__(self, name:str, run_mode:RunMode,
                 ch_in:Optional[int]=None, ch_out:Optional[int]=None,
                stride:Optional[int]=None, affine:Optional[bool]=None)->None:
        self.name = name
        self.ch_in, self.ch_out = ch_in, ch_out
        self.stride, self.affine, self.run_mode = stride, affine, run_mode

class EdgeDesc(DescBase):
    """Edge description between two nodes in the cell
    """
    def __init__(self, op_desc:OpDesc, index:int, input_ids:List[int],
                 from_node:int, to_state:int)->None:
        self.op_desc = op_desc
        self.index = index
        self.input_ids = input_ids
        self.from_node = from_node
        self.to_state = to_state

class NodeDesc(DescBase):
    def __init__(self, edges:List[EdgeDesc]=[]) -> None:
        self.edges = edges

class AuxTowerDesc(DescBase):
    def __init__(self, ch_in:int, n_classes:int, aux_weight:float) -> None:
        self.ch_in = ch_in
        self.n_classes = n_classes
        self.aux_weight = aux_weight

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class CellDesc(DescBase):
    """Cell description
    """
    def __init__(self, cell_type:CellType, index:int, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc, aux_tower_desc:Optional[AuxTowerDesc],
            n_out_nodes:int, n_node_channels:int,
            alphas_from:int, run_mode:RunMode)->None:
        self.cell_type = cell_type
        self.index = index
        self.nodes = nodes
        self.s0_op = s0_op
        self.s1_op = s1_op
        self.aux_tower_desc = aux_tower_desc
        self.n_out_nodes = n_out_nodes
        self.n_node_channels = n_node_channels
        self.run_mode = run_mode
        self.alphas_from = alphas_from

    def get_ch_out(self)->int:
        # cell output is concatenation of output nodes and output channels
        return self.n_out_nodes * self.n_node_channels

class ModelDesc(DescBase):
    def __init__(self, stem0_op:OpDesc, stem1_op:OpDesc, pool_op:OpDesc,
                 ch_in:int, n_classes:int, cell_descs:List[CellDesc])->None:
        self.stem0_op, self.stem1_op, self.pool_op = stem0_op, stem1_op, pool_op
        self.ch_in = ch_in
        self.n_classes = n_classes
        self.cell_descs = cell_descs

