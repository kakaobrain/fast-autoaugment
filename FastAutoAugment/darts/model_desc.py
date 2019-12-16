from enum import Enum
from typing import Optional, List, Sequence, Final
import yaml

class DescBase:
    def serialize(self)->dict:
        return yaml.dump(self)
    def deserialize(self, v:dict)->'DescBase':
        return yaml.load(v)

class OpDesc(DescBase):
    """Op description that is in each edge
    """
    def __init__(self, name:str, training:bool,
                 ch_in:Optional[int]=None, ch_out:Optional[int]=None,
                stride:Optional[int]=None, affine:Optional[bool]=None)->None:
        self.name = name
        self.ch_in, self.ch_out = ch_in, ch_out
        self.stride, self.affine, self.training = stride, affine, training

class EdgeDesc(DescBase):
    """Edge description between two nodes in the cell
    """
    def __init__(self, op_desc:OpDesc, input_ids:List[int],
                 from_node:int, to_state:int)->None:
        self.op_desc = op_desc
        self.input_ids = input_ids
        self.from_node = from_node
        self.to_state = to_state

class NodeDesc(DescBase):
    def __init__(self, edges:List[EdgeDesc]=[]) -> None:
        self.edges = edges

class AuxTowerDesc(DescBase):
    def __init__(self, n_classes:int, aux_weight:float) -> None:
        self.n_classes = n_classes
        self.aux_weight = aux_weight

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class CellDesc(DescBase):
    """Cell description
    """
    def __init__(self, cell_type:CellType, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc, aux_tower_desc:Optional[AuxTowerDesc],
            n_out_nodes:int, n_node_channels:int,
            alphas_from:int, training:bool)->None:
        self.cell_type = cell_type
        self.nodes = nodes
        self.s0_op = s0_op
        self.s1_op = s1_op
        self.aux_tower_desc = aux_tower_desc
        self.n_out_nodes = n_out_nodes
        self.n_node_channels = n_node_channels
        self.training = training
        self.alphas_from = alphas_from

    def get_ch_out(self)->int:
        # cell output is concate of output nodes and output channels
        return self.n_out_nodes * self.n_node_channels

class ModelDesc(DescBase):
    def __init__(self, stem0_op:OpDesc, stem1_op:OpDesc, pool_op:OpDesc,
                 ch_in:int, n_classes:int, cell_descs:List[CellDesc])->None:
        self.stem0_op, self.stem1_op, self.pool_op = stem0_op, stem1_op, pool_op
        self.ch_in = ch_in
        self.n_classes = n_classes
        self.cell_descs = cell_descs

def darts_model_desc(ch_in:int, n_classes:int,
        stem0_op_name:str, stem1_op_name:str, pool_op_name:str,
        n_cells:int, n_nodes:int, n_out_nodes:int, init_ch_out:int,
        stem_multiplier:int, aux_weight:float, training:bool)->ModelDesc:

    # TODO: weired not always use two different stemps as in original code
    # TODO: why do we need stem_multiplier?
    stem_ch_out=init_ch_out*stem_multiplier
    stem0_op = OpDesc(name=stem0_op_name, training=training,
                     ch_in=ch_in, ch_out=stem_ch_out, affine=True)
    stem1_op = OpDesc(name=stem1_op_name, training=training,
                      ch_in=ch_in, ch_out=stem_ch_out, affine=True)

    cell_descs = []
    reduction_p = False
    first_normal, first_reduction = -1, -1

    pp_ch_out, p_ch_out, ch_out = stem_ch_out, stem_ch_out, init_ch_out

    for ci in range(n_cells):
        reduction = (ci+1)%3==0
        if reduction:
            ch_out, cell_type = ch_out*2, CellType.Reduction
        else:
            cell_type = CellType.Regular

        # TODO: investigate why affine=False for search but True for test
        if reduction_p:
            s0_op = OpDesc('prepr_reduce', training=training,
                           ch_in=pp_ch_out, ch_out=ch_out, affine=False)
        else:
            s0_op = OpDesc('prepr_normal', training=training,
                        ch_in=pp_ch_out, ch_out=ch_out, affine=not training)
        s1_op = OpDesc('prepr_normal', training=training,
                       ch_in=p_ch_out, ch_out=ch_out, affine=not training)

        if cell_type == CellType.Regular:
            first_normal = ci if first_normal < 0 else first_normal
            alphas_from = first_normal
        elif cell_type == CellType.Reduction:
            first_reduction = ci if first_reduction < 0 else first_reduction
            alphas_from = first_reduction
        else:
            raise NotImplementedError(f'CellType {cell_type} is not implemented')

        nodes:List[NodeDesc] = []
        for i in range(n_nodes):
            edges:List[EdgeDesc] = []
            for j in range(i+2):
                op_desc = OpDesc('mixed_op',
                                training=training,
                                ch_in=ch_out,
                                ch_out=ch_out,
                                stride=2 if reduction and j < 2 else 1,
                                affine=not training)
                edge = EdgeDesc(op_desc,
                                input_ids=[j],
                                from_node=i,
                                to_state=j)
                edges.append(edge)
            nodes.append(NodeDesc(edges=edges))

        aux_tower_desc = None
        # TODO: shouldn't we adding aux tower at *every* 1/3rd?
        if not training and aux_weight > 0. and ci==2*n_cells//3:
            aux_tower_desc = AuxTowerDesc(n_classes, aux_weight)

        cell_descs.append(CellDesc(
            cell_type=cell_type,
            nodes=[NodeDesc() for _ in range(n_nodes)],
            s0_op=s0_op, s1_op=s1_op, aux_tower_desc=aux_tower_desc,
            n_out_nodes=n_out_nodes,
            n_node_channels=ch_out,
            alphas_from=alphas_from,
            training=training
        ))

        # we concate all channels so next cell node gets channels from all nodes
        pp_ch_out, p_ch_out = p_ch_out, cell_descs[-1].get_ch_out()
        reduction_p = reduction

    ch_out = cell_descs[-1].get_ch_out()
    pool_op = OpDesc(pool_op_name, training, ch_out, ch_out)

    return ModelDesc(stem0_op, stem1_op, pool_op, ch_in, n_classes, cell_descs)

