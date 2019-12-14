from typing import Iterator, List, Sequence, Optional

import torch
from torch import nn
from torch.nn.modules.container import ModuleList

from .operations import OPS, FactorizedReduce, ReLUConvBN, MixedOp

class SearchCellEdge(nn.Module):
    def __init__(self, op:nn.Module, input_ids:List[int],
                 alphas:Optional[nn.Parameter])->None:
        super().__init__()
        self._op = op
        self._alphas = alphas
        self._input_ids = input_ids

    def forward(self, inputs:List[torch.Tensor]):
        if len(self._input_ids)==1:
            return self._op(inputs[self._input_ids[0]])
        elif len(self._input_ids) == len(inputs): # for perf
            return self._op(inputs)
        else:
            return self._op([inputs[i] for i in self._input_ids])

    def get_alphas(self)->Optional[nn.Parameter]:
        return self._alphas


class DartCellEdge(SearchCellEdge):
    def __init__(self, ch_out:int, state_id:int, reduction:bool,
                 alphas_edge:Optional['DartCellEdge'])->None:
        op_alphas = None if alphas_edge is None else alphas_edge.get_alphas()
        # reduction should be used only for first 2 input node
        stride = 2 if reduction and state_id < 2 else 1
        op = MixedOp(ch_out, stride, op_alphas)

        super().__init__(op, [state_id], op.get_alphas())


class SearchCell(nn.Module):
    def __init__(self, n_nodes: int, n_node_outs: int, ch_pp: int, ch_p: int,
                 ch_out:int, reduction: bool, reduction_prev: bool,
                 first_cell:Optional['SearchCell']):
        """
        A cell k takes input from last two cells k-2, k-1. The cell consists
        of `n_nodes` so that on each node i, we take output of all previous i
        nodes + 2 cell inputs, apply op on each of these outputs and produce
        their sum as output of i-th node. Each op output has ch_out channels.
        The output of the cell produced by forward() is concatenation of last
        `n_node_outs` number of nodes. _Cell could be a reduction cell or it
        could be a normal cell. The diference between two is that reduction
        cell uses stride=2 for the ops that connects to cell inputs.

        :param n_nodes: 4, number of nodes inside a cell
        :param n_node_outs: 4, number of last nodes to concatenate as output,
            this will multiply number of channels in node
        :param ch_pp: 48, channels from cell k-2
        :param ch_p: 48, channels from cell k-1
        :param ch_out: 16, output channels for each node
        :param reduction: Is this reduction cell? If so reduce output size
        :param reduction_prev: Was previous cell reduction? Is so we should
            resize reduce the s0 width by half.
        """
        super().__init__()

        # indicating current cell is reduction or not
        self.reduction = reduction

        """We get output from cells i-1 and i-2.
        If i-1 was reduction cell then output shapes of i-1 and i-2 don't match.
        In tha case we reduce i-1 output by 4X as well.
        If i-2 was reduction cell then i-1 and i-2 output will match."""
        # TODO: reduction cell might have output reduced by 2^1=2X due to
        #   stride 2 through input nodes however FactorizedReduce does only
        #   4X reduction. Is this correct?
        if reduction_prev:
            self._preprocess0 = FactorizedReduce(ch_pp, ch_out, affine=False)
        else:  # use 1x1 conv to get desired channels
            self._preprocess0 = ReLUConvBN(
                ch_pp, ch_out, 1, 1, 0, affine=False)
        # _preprocess1 deal with output from prev cell
        self._preprocess1 = ReLUConvBN(ch_p, ch_out, 1, 1, 0, affine=False)

        # n_nodes inside a cell
        self.n_nodes = n_nodes  # 4
        self.n_node_outs = n_node_outs  # 4

        # dag has n_nodes, each node is list containing edges to previous nodes
        # Each edge in dag is populated with MixedOp but it could
        # be some other op as well
        self._dag:nn.ModuleList = nn.ModuleList()

        for i in range(self.n_nodes):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            edges: nn.ModuleList = nn.ModuleList()
            self._dag.append(edges)
            for j in range(2 + i):  # include 2 input nodes
                edge = DartCellEdge(ch_out, j, reduction,
                    None if first_cell is None else first_cell._dag[i][j])
                edges.append(edge)

    def get_alphas(self)->Iterator[nn.Parameter]:
        for edges in self._dag:
            for edge in edges:
                yield edge.get_alphas()

    def forward(self, s0, s1):
        """

        :param s0: output of cell k-1
        :param s1: output of cell k-2
        :param alphas_sm: List of alphas for each cell with softmax applied
        """

        # print('s0:', s0.shape,end='=>')
        s0 = self._preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
        s1 = self._preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]

        node_outs = [s0, s1]

        # for each node, receive input from all previous nodes and s0, s1
        node_alphas: nn.Parameter  # shape (i+2, n_ops)
        edges:List['SearchCellEdge']

        for edges in self._dag:
            # why do we do sum? Hope is that some weight will win and others
            # would lose masking their outputs
            # TODO: we should probably do average here otherwise output will
            #   blow up as number of primitives grows
            o = sum(edges[i](o) for i,o in enumerate(node_outs))
            # append one state since s is the elem-wise addition of all output
            node_outs.append(o)

        # concat along dim=channel
        # TODO: Below assumes same shape except for channels but this won't
        #   happen for max pool etc shapes?
        # 6x[40,16,32,32]
        return torch.cat(node_outs[-self.n_node_outs:], dim=1)
