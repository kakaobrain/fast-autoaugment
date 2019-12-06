import torch
from torch import nn
from torch.nn import modules
import torch.nn.functional as F
from torch.nn.modules.module import Module
import logging

from .operations import OPS, FactorizedReduce, ReLUConvBN, MixedOp
from . import genotypes as gt

class _Cell(nn.Module):

    def __init__(self, n_nodes:int, n_node_outs:int, ch_pp:int, ch_p:int,
        ch_out:int, reduction:bool, reduction_prev:bool):
        """
        Each cell k takes input from last two cells k-2, k-1. The cell consists
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
        super(_Cell, self).__init__()

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
        else: # use 1x1 conv to get desired channels
            self._preprocess0 = ReLUConvBN(ch_pp, ch_out, 1, 1, 0, affine=False)
        # _preprocess1 deal with output from prev cell
        self._preprocess1 = ReLUConvBN(ch_p, ch_out, 1, 1, 0, affine=False)

        # n_nodes inside a cell
        self.n_nodes = n_nodes # 4
        self.n_node_outs = n_node_outs # 4

        # dag has n_nodes, each node is list containing edges to previous nodes
        # Each edge in dag is populated with MixedOp but it could
        # be some other op as well
        self._dag = nn.ModuleList()

        for i in range(self.n_nodes):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            self._dag.append(nn.ModuleList())
            for j in range(2 + i): # include 2 input nodes
                # reduction should be used only for first 2 input node
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(ch_out, stride)
                self._dag[i].append(op)

    def forward(self, s0, s1, alphas_sm):
        """

        :param s0: output of cell k-1
        :param s1: output of cell k-2
        :param alphas_sm: List of alphas for each cell with softmax applied
        """

        # print('s0:', s0.shape,end='=>')
        s0 = self._preprocess0(s0) # [40, 48, 32, 32], [40, 16, 32, 32]
        s1 = self._preprocess1(s1) # [40, 48, 32, 32], [40, 16, 32, 32]

        node_outs = [s0, s1]

        # for each node, receive input from all previous nodes and s0, s1
        node_alphas:nn.Parameter # shape (i+2, n_ops)
        node_ops:nn.ModuleList # list of MixedOp operating on previous nodes

        for node_ops, node_alphas in zip(self._dag, alphas_sm):
            # take each previous ouput and column of node_alphas param
            # (each column has n_ops trainable params)
            out_alpha = zip(node_outs, node_alphas)

            # why do we do sum? Hope is that some weight will win and others
            # would lose masking their outputs
            # TODO: we should probably do average here otherwise output will
            #   blow up as number of primitives grows
            o = sum(node_ops[i](o, w) for i, (o, w) in enumerate(out_alpha))
            # append one state since s is the elem-wise addition of all output
            node_outs.append(o)

        # concat along dim=channel
        # TODO: Below assumes same shape except for channels but this won't
        #   happen for max pool etc shapes?
        return torch.cat(node_outs[-self.n_node_outs:], dim=1) # 6x[40,16,32,32]

class _CnnModel(nn.Module):
    """ Search CNN model """

    def __init__(self, ch_in:int, ch_out_init:int, n_classes:int, n_layers:int,
        n_nodes=4, n_node_outs=4, stem_multiplier=3):
        """

        :param ch_in: number of channels in input image (3)
        :param ch_out_init: number of output channels from the first layer) /
            stem_multiplier (16)
        :param n_classes: number of classes
        :param n_layers: number of cells of current network
        :param n_nodes: nodes inside cell
        :param n_node_outs: output channel of cell = n_node_outs * ch
        :param stem_multiplier: output channel of stem net = stem_multiplier*ch
        """
        super().__init__()

        self.ch_in = ch_in
        self.ch_out_init = ch_out_init
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_node_outs = n_node_outs

        # stem is the start of network. This is additional
        # 3x3 conv layer that multiplies channels
        # TODO: why do we need stem_multiplier?
        ch_cur = stem_multiplier * ch_out_init # 3*16
        self.stem = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to
            # BN in conv layer.
            nn.Conv2d(ch_in, ch_cur, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_cur)
        )

        # ch_cur: output channels for cell i
        # ch_p: output channels for cell i-1
        # ch_pp: output channels for cell i-2
        ch_pp, ch_p, ch_cur = ch_cur, ch_cur, ch_out_init # 48, 48, 16
        self._cells = nn.ModuleList()
        reduction_prev = False
        for i in range(n_layers):
            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [n_layers // 3, 2 * n_layers // 3]:
                ch_cur, reduction = ch_cur * 2, True
            else:
                reduction = False

            # [ch_p, h, h] => [n_node_outs*ch_cur, h/h//2, h/h//2]
            # the output channels = n_node_outs * ch_cur
            cell = _Cell(n_nodes, n_node_outs, ch_pp, ch_p, ch_cur, reduction,
                reduction_prev)
            # update reduction_prev
            reduction_prev = reduction

            self._cells.append(cell)

            ch_pp, ch_p = ch_p, n_node_outs * ch_cur

        # adaptive pooling output size to 1x1
        self.final_pooling = nn.AdaptiveAvgPool2d(1)
        # since ch_p records last cell's output channels
        # it indicates the input channel number
        self.linear = nn.Linear(ch_p, n_classes)

    def forward(self, x, alphas_sm_normal, alphas_sm_reduce):
        """
        Runs x through cells with alphas, applies final pooling, send through
            FCs and returns logits.

        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        """

        # first two inputs
        s0 = s1 = self.stem(x) # [b, 3, 32, 32] => [b, 48, 32, 32]
        # macro structure: each cell consumes output of
        # previous two cells
        for cell in self._cells:
            alphas_sm = alphas_sm_reduce if cell.reduction else alphas_sm_normal
            s0, s1 = s1, cell(s0, s1, alphas_sm) # [40, 64, 32, 32]

        # s1 is now the last cell's output
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1)) # flatten

        return logits

class CnnArchModel(nn.Module):
    def __init__(self, ch_in, ch_out_init, n_classes, n_layers, criterion,
            n_nodes=4, n_node_outs=4, stem_multiplier=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion

        # alphas must be created before we create inner model
        self._create_alpahs()

        self._model = _CnnModel(ch_in, ch_out_init, n_classes, n_layers,
            n_nodes, n_node_outs, stem_multiplier)

    def _create_alpahs(self):
        # create alpha parameters for each node.
        # Alphas are shared between cells so each cell has identical topology.

        n_ops = len(gt.PRIMITIVES)

        self._alphas_normal = nn.ParameterList()
        self._alphas_reduce = nn.ParameterList()
        self._alphas = [] # all alpha params for faster access

        # TODO: is unofrm rand init good idea?
        for i in range(self.n_nodes):
            self._alphas_normal.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self._alphas_reduce.append(
                nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n: # TODO: may be directly add above params?
                self._alphas.append((n, p))

    def forward(self, x):
        # pass weights through softmax, squashing them between 0 to 1
        alphas_sm_normal = [F.softmax(alpha, dim=-1)
            for alpha in self._alphas_normal]
        alphas_sm_reduce = [F.softmax(alpha, dim=-1)
            for alpha in self._alphas_reduce]

        return self._model(x, alphas_sm_normal, alphas_sm_reduce)

    def loss(self, x, target):
        logits = self(x)
        return self.criterion(logits, target)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self._alphas_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self._alphas_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self._alphas_normal, k=2)
        gene_reduce = gt.parse(self._alphas_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self._model.parameters()

    def named_weights(self):
        return self._model.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
