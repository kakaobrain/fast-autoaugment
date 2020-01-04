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
    def __init__(self, name:str, params:dict={}, in_len=1)->None:
        self.name = name
        self.in_len = in_len
        self.params = params # parameters specific to op needed to construct it

class EdgeDesc(DescBase):
    """Edge description between two nodes in the cell
    """
    def __init__(self, op_desc:OpDesc, index:int, input_ids:List[int],
                 run_mode:RunMode)->None:
        assert op_desc.in_len == len(input_ids)
        self.op_desc = op_desc
        self.index = index
        self.input_ids = input_ids
        self.run_mode = run_mode

class NodeDesc(DescBase):
    def __init__(self, edges:List[EdgeDesc]=[]) -> None:
        self.edges = edges

class AuxTowerDesc(DescBase):
    def __init__(self, ch_in:int, n_classes:int) -> None:
        self.ch_in = ch_in
        self.n_classes = n_classes

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class ConvMacroParams:
    """Holds parameters that may be altered by macro architecture"""

    def __init__(self, ch_in:int, ch_out:int, affine:bool) -> None:
        self.ch_in, self.ch_out = ch_in, ch_out
        self.affine = affine

class CellDesc(DescBase):
    def __init__(self, cell_type:CellType, index:int, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc,
            out_nodes:int, node_ch_out:int,
            alphas_from:int, run_mode:RunMode)->None:
        assert s0_op.params['conv'].ch_out == s1_op.params['conv'].ch_out
        assert s0_op.params['conv'].ch_out == node_ch_out

        self.cell_type = cell_type
        self.index = index
        self.nodes = nodes
        self.s0_op, self.s1_op = s0_op, s1_op
        self.out_nodes, self.node_ch_out = out_nodes, node_ch_out
        self.run_mode = run_mode
        self.alphas_from = alphas_from # cell index with which we share alphas

        self.cell_ch_out = out_nodes * node_ch_out
        self.conv_params = ConvMacroParams(node_ch_out,
                                           node_ch_out,
                                           run_mode!=RunMode.Search)

class ModelDesc(DescBase):
    def __init__(self, stem0_op:OpDesc, stem1_op:OpDesc, pool_op:OpDesc,
                 ds_ch:int, n_classes:int, cell_descs:List[CellDesc],
                 aux_tower_descs:List[Optional[AuxTowerDesc]])->None:
        assert len(cell_descs) == len(aux_tower_descs)
        self.stem0_op, self.stem1_op, self.pool_op = stem0_op, stem1_op, pool_op
        self.ds_ch = ds_ch
        self.n_classes = n_classes
        self.cell_descs = cell_descs
        self.aux_tower_descs = aux_tower_descs

