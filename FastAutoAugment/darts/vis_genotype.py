""" Network architecture visualizer using graphviz """
import sys
from graphviz import Digraph
from typing import Union, List, Tuple, Optional

from .model_desc import CellDesc, CellType, ModelDesc
from ..common.utils import first_or_default

def draw_model_desc(model_desc:ModelDesc, file_path:str=None, caption:str=None,
                    render=True)->Tuple[Optional[Digraph],Optional[Digraph]]:
    normal_cell_desc = first_or_default((c for c in model_desc.cell_descs \
                                        if c.cell_type == CellType.Regular), None)

    reduced_cell_desc = first_or_default((c for c in model_desc.cell_descs \
                                        if c.cell_type == CellType.Reduction), None)

    g_normal = draw_cell_desc(normal_cell_desc, file_path, caption, render) \
            if normal_cell_desc is not None else None
    g_reduct = draw_cell_desc(reduced_cell_desc, file_path, caption, render) \
            if reduced_cell_desc is not None else None

    return g_normal, g_reduct

def draw_cell_desc(cell_desc:CellDesc, file_path:str=None, caption:str=None,
                   render=True)->Digraph:
    """ make DAG plot and optionally save to file_path as .png """

    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    # TODO: remove only two input node as assumption
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(cell_desc.nodes)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, node in enumerate(cell_desc.nodes):
        for edge in node.edges:
            op, js = edge.op_desc.name, edge.input_ids
            for j in js:
                if j == 0:
                    u = "c_{k-2}"
                elif j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j-2)

                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    if render:
        g.render(file_path, view=False)
    return g
