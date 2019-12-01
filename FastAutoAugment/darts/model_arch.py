import  torch
from    torch import nn
import  torch.nn.functional as F

from    .operations import OPS, FactorizedReduce, ReLUConvBN, MixedOp
from    .genotypes import PRIMITIVES, Genotype

class Cell(nn.Module):

    def __init__(self, n_nodes, n_out_nodes, cpp, cp, ch_out, reduction, reduction_prev):
        """
        Each cell k takes input from last two cells k-2, k-1. The cell consists of `n_nodes` so that on each node i,
        we take output of all previous i nodes + 2 cell inputs, apply op on each of these outputs and produce their
        sum as output of i-th node.
        Each op output has ch_out channels. The output of the cell produced by forward() is concatenation of last
        `n_out_nodes` number of nodes. Cell could be a reduction cell or it could be a normal cell. The only
        diference between two is that reduction cell uses stride=2 for the ops that connects to cell inputs.

        :param n_nodes: 4, number of nodes inside a cell
        :param n_out_nodes: 4, number of last nodes to concatenate as output, this will multiply number of channels in node
        :param cpp: 48, channels from cell k-2
        :param cp: 48, channels from cell k-1
        :param ch_out: 16, output channels for each node
        :param reduction: indicates whether to reduce the output maps width
        :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        """
        super(Cell, self).__init__()

        # indicating current cell is reduction or not
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            # if prev cell has reduced channel/double width,
            # it will reduce width by half
            self.preprocess0 = FactorizedReduce(cpp, ch_out, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(cpp, ch_out, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(cp, ch_out, 1, 1, 0, affine=False)

        # n_nodes inside a cell
        self.n_nodes = n_nodes # 4
        self.n_out_nodes = n_out_nodes # 4

        self.dag = nn.ModuleList()

        for i in range(self.n_nodes):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(ch_out, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, arch_weights):
        """

        :param s0: output of cell k-1
        :param s1: output of cell k-2
        :param weights: [14, 8], weights for primitives for each edge
        :return:
        """
        # print('s0:', s0.shape,end='=>')
        s0 = self.preprocess0(s0) # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s0.shape, self.reduction_prev)
        # print('s1:', s1.shape,end='=>')
        s1 = self.preprocess1(s1) # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s1.shape)

        states = [s0, s1]
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for edges, w_list in zip(self.dag, arch_weights):
            s = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            # append one state since s is the elem-wise addition of all output
            states.append(s)

        # concat along dim=channel
        return torch.cat(states[-self.n_out_nodes:], dim=1) # 6 of [40, 16, 32, 32]


class Network(nn.Module):

    """
    stack number:layer of cells and then flatten to fed a linear layer
    """
    def __init__(self, ch_in, ch_out_init, n_classes, n_layers, criterion, n_nodes=4, n_out_nodes=4, stem_multiplier=3):
        """

        :param ch_in: 3 (number of channels in input image)
        :param ch_out_init: 16 (number of output channels from the first layer) / stem_multiplier
        :param n_classes: number of classes
        :param n_layers: number of cells of current network
        :param criterion:
        :param n_nodes: nodes inside cell
        :param n_out_nodes: output channel of cell = n_out_nodes * ch
        :param stem_multiplier: output channel of stem net = stem_multiplier * ch
        """
        super(Network, self).__init__()

        self.ch_in = ch_in
        self.ch_out_init = ch_out_init
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.criterion = criterion
        self.n_nodes = n_nodes
        self.n_out_nodes = n_out_nodes

        # stem_multiplier is for stem network,
        # and n_out_nodes is for general cell
        # TODO: why do we need stem_multiplier?
        c_curr = stem_multiplier * ch_out_init # 3*16
        # stem network, convert 3 channel to c_curr
        # First layer is 3x3, stride 1 layer and assumes 3 channel input
        self.stem = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to this in conv layer.
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr)
        )

        # c_curr means a factor of the output channels of current cell
        # output channels = n_out_nodes * c_curr
        # c_curr: current cell output channels
        # cp: previous cell output channels
        # cpp: previous previous cell output channels
        cpp, cp, c_curr = c_curr, c_curr, ch_out_init # 48, 48, 16
        self._cells = nn.ModuleList()
        reduction_prev = False
        for i in range(n_layers):
            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            # [cp, h, h] => [n_out_nodes*c_curr, h/h//2, h/h//2]
            # the output channels = n_out_nodes * c_curr
            cell = Cell(n_nodes, n_out_nodes, cpp, cp, c_curr, reduction, reduction_prev)
            # update reduction_prev
            reduction_prev = reduction

            self._cells.append(cell)

            cpp, cp = cp, n_out_nodes * c_curr

        # adaptive pooling output size to 1x1
        self.final_pooling = nn.AdaptiveAvgPool2d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        self.linear = nn.Linear(cp, n_classes)

        # k is the total number of edges inside single cell, 14
        k = sum(1 for i in range(self.n_nodes) for j in range(2 + i))
        n_ops = len(PRIMITIVES) # 8

        # create k*n_ops parameters that we will share between all cells
        # this kind of implementation will add alpha into self.parameters()
        # it has num k of alpha parameters, and each alpha shape: [n_ops]
        # it requires grad and can be converted to cpu/gpu automatically
        self.alpha_normal = nn.Parameter(torch.randn(k, n_ops))
        self.alpha_reduce = nn.Parameter(torch.randn(k, n_ops))
        with torch.no_grad():
            # initialize to smaller value
            self.alpha_normal.mul_(1e-3)
            self.alpha_reduce.mul_(1e-3)
        self._arch_parameters = [
            self.alpha_normal,
            self.alpha_reduce,
        ]

    def new(self):
        """
        create a new model and initialize it with current alpha parameters.
        However, its weights are left at initial value.
        :return:
        """
        model_new = Network(self.ch_in, self.ch_out_init, self.n_classes, self.n_layers, self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        """
        Runs x through cells, applies final pooling, send through FCs and returns logits.
        This would tech alpha parameters into account.

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
        :param x:
        :return:
        """
        # s0 & s1 means the last cells' output
        s0 = s1 = self.stem(x) # [b, 3, 32, 32] => [b, 48, 32, 32]

        for i, cell in enumerate(self._cells):
            # Arch weights are shared across cells according to current cell's type.
            # Softmax will squash all weights to 0 to 1 making them differentiable switches.
            if cell.reduction:
                weights = F.softmax(self.alpha_reduce, dim=-1)
            else:
                weights = F.softmax(self.alpha_normal, dim=-1) # [14, 8]
            # execute cell() firstly and then assign s0=s1, s1=result
            s0, s1 = s1, cell(s0, s1, weights) # [40, 64, 32, 32]

        # s1 is the last cell's output
        out = self.final_pooling(s1)
        logits = self.linear(out.view(out.size(0), -1))

        return logits

    def loss(self, x, target):
        """

        :param x:
        :param target:
        :return:
        """
        logits = self(x)
        return self.criterion(logits, target)



    def arch_parameters(self):
        return self._arch_parameters




    def genotype(self):
        """
        Returns display description of network from the weights for each ops in cell

        :return:
        """
        def _parse(weights):
            """
            We have 4 nodes, each step can have edge with previous nodes + 2 inputs.
            So total edges = 2 + 3 + 4 + 5 = 14
            We will have total 8 primitives for each of the 14 edges within each cell.
            So weight shape is [14, 8]. These are the alpha parameters and shared across cells.
            For each of the edges for a node, we want to find out top 2 strongest prmitives
            and make them as "final" for that node. As we don't consider none edge,
            this guerentees that each node will exactly end up with 2 edges, one final non-none
            primitive attached to each.

            :param weights: [14, 8]
            :return: string array, each member describes edge in the cell
            """
            gene = []
            n = 2
            start = 0
            for i in range(self.n_nodes): # for each node
                end = start + n
                W = weights[start:end].copy() # [2, 8], [3, 8], ...

                # so far each edge has 8 primitives attached, we will chose top two so that
                # we get two best edges on which best primitives resides
                edges = sorted(range(i + 2), # i+2 is the number of connection for node i
                            key=lambda x: -max(W[x][k] # by descending order
                                               for k in range(len(W[x])) # get strongest ops
                                               if k != PRIMITIVES.index('none'))
                               )[:2] # only has two inputs

                # Each node i now we have 2 edges that still has 8 primitives each and
                # we want to select best primitive for each edge
                for j in edges: # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])): # get strongest ops for current input j->i
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j)) # save ops and input node
                start = end
                n += 1

            # at this point we should have each node, with exactly two edges and associated best primitive
            return gene

        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.n_nodes - self.n_out_nodes, self.n_nodes + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype
