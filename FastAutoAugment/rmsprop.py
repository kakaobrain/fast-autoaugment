import torch
from torch.optim.optimizer import Optimizer


class RMSpropTF(Optimizer):
    r"""Implements RMSprop algorithm.
    Reimplement original formulation to match TF rmsprop
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v + \epsilon})` where :math:`\alpha` from :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, momentum=0, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        assert momentum > 0.0

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['ms'] = torch.ones_like(p.data)  #, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['mom'] = torch.zeros_like(p.data)  #, memory_format=torch.preserve_format)

                # weight decay -----
                grad = grad.add(group['weight_decay'], p.data)

                ms = state['ms']
                rho = group['alpha']
                mom = state['mom']
                state['step'] += 1

                # ms.mul_(rho).addcmul_(1 - rho, grad, grad)
                # ms.addcmul_(1 - rho, torch.mul(grad, grad), grad)
                ms.add_(torch.mul(grad, grad).add_(-ms) * (1. - rho))
                mom.mul_(group['momentum']).addcdiv_(grad * group['lr'], (ms + group['eps']).sqrt())
                p.data.add_(-1.0, mom)

        return loss
