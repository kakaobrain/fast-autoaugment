import torch
from torch.nn import BatchNorm2d
from torch.nn.parameter import Parameter
import torch.distributed as dist
from torch import nn


class TpuBatchNormalization(nn.Module):
    # Ref : https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py#L113
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(TpuBatchNormalization, self).__init__()   # num_features, eps, momentum, affine, track_running_stats)

        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.eps = eps
        self.momentum = momentum

    def _reduce_avg(self, t):
        dist.all_reduce(t, dist.ReduceOp.SUM)
        t.mul_(1. / dist.get_world_size())

    def forward(self, input):
        if not self.training or not dist.is_initialized():
            bn = (input - self.running_mean.view(1, self.running_mean.shape[0], 1, 1)) / \
                 (torch.sqrt(self.running_var.view(1, self.running_var.shape[0], 1, 1) + self.eps))
            # print(self.weight.shape, self.bias.shape)
            return bn.mul(self.weight.view(1, self.weight.shape[0], 1, 1)).add(self.bias.view(1, self.bias.shape[0], 1, 1))

        shard_mean, shard_invstd = torch.batch_norm_stats(input, self.eps)
        shard_vars = (1. / shard_invstd) ** 2 - self.eps

        shard_square_of_mean = torch.mul(shard_mean, shard_mean)
        shard_mean_of_square = shard_vars + shard_square_of_mean

        group_mean = shard_mean.clone().detach()
        self._reduce_avg(group_mean)
        group_mean_of_square = shard_mean_of_square.clone().detach()
        self._reduce_avg(group_mean_of_square)
        group_vars = group_mean_of_square - torch.mul(group_mean, group_mean)

        group_mean = group_mean.detach()
        group_vars = group_vars.detach()

        # print(self.running_mean.shape, self.running_var.shape)
        self.running_mean.mul_(1. - self.momentum).add_(group_mean.mul(self.momentum))
        self.running_var.mul_(1. - self.momentum).add_(group_vars.mul(self.momentum))
        self.num_batches_tracked.add_(1)

        # print(input.shape, group_mean.view(1, group_mean.shape[0], 1, 1).shape, group_vars.view(1, group_vars.shape[0], 1, 1).shape, self.eps)
        bn = (input - group_mean.view(1, group_mean.shape[0], 1, 1)) / (torch.sqrt(group_vars.view(1, group_vars.shape[0], 1, 1) + self.eps))
        # print(self.weight.shape, self.bias.shape)
        return bn.mul(self.weight.view(1, self.weight.shape[0], 1, 1)).add(self.bias.view(1, self.bias.shape[0], 1, 1))
