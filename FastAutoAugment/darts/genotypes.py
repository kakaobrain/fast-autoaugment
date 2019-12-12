""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""

from collections import namedtuple
import torch
import torch.nn as nn
from typing import List

# TODO: remove circular reference
from . import operations as ops

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', #identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none' # this must be at the end of the list (req by parse())
]

NASNet = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
        [('avg_pool_3x3', 1), ('skip_connect', 0)],
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 0)],
        [('sep_conv_3x3', 1), ('skip_connect', 1)],
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_7x7', 0)],
        [('max_pool_3x3', 1), ('sep_conv_7x7', 0)],
        [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)],
        [('skip_connect', 3), ('avg_pool_3x3', 2)],
        [('sep_conv_3x3', 2), ('max_pool_3x3', 1)],
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[[('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)],
            [('sep_conv_3x3', 0), ('avg_pool_3x3', 3)],
            [('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 0), ('avg_pool_3x3', 1)],
    ],
    normal_concat=[4, 5, 6],
    reduce=[[('avg_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_7x7', 2)],
            [('sep_conv_7x7', 0), ('avg_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('conv_7x1_1x7', 0), ('sep_conv_3x3', 5)],
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('skip_connect', 2)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2)],
            [('skip_connect', 2), ('avg_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 1), ('skip_connect', 0)],
            [('skip_connect', 0), ('dil_conv_3x3', 2)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 2)],
            [('skip_connect', 2), ('max_pool_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

MyDARTS = Genotype(
    normal=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 2)],
            [('max_pool_3x3', 2), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 2), ('max_pool_3x3', 0)]],
    normal_concat=range(2, 6),
    reduce=[[('dil_conv_5x5', 1), ('dil_conv_5x5', 0)],
            [('max_pool_3x3', 2), ('sep_conv_5x5', 1)],
            [('dil_conv_5x5', 3), ('sep_conv_5x5', 0)],
            [('dil_conv_5x5', 2), ('sep_conv_5x5', 3)]],
    reduce_concat=range(2, 6))

PT_DARTS = Genotype(
    normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)],
            [('skip_connect', 0), ('dil_conv_3x3', 2)],
            [('sep_conv_3x3', 1), ('skip_connect', 0)],
            [('sep_conv_3x3', 1), ('skip_connect', 0)]],
    normal_concat=range(2, 6),
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 2)],
            [('skip_connect', 3), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 0)]],
    reduce_concat=range(2, 6)
)

# This is the genotype we would find and is default for testing
DARTS = DARTS_V1


def to_dag(ch_in:int, gene:List[tuple], reduction:bool)->nn.ModuleList:
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](ch_in, stride, True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s)->Genotype:
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(alpha, k)->List[tuple]:
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]

    For CNNs:
    We have 4 nodes, each can have edge with previous nodes + 2 inputs.
    Total edges = 2 + 3 + 4 + 5 = 14
    We will have total 8 primitives for each of the 14 edges within each cell.
    For each of the edges for a node, we want to find out top 2 strongest prmitives
    and make them as "final" for that node. As we don't consider none edge,
    this guerentees that each node will exactly end up with 2 edges, one final non-none
    primitive attached to each.

    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene
