from typing import Optional, Tuple, Union
import os
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides

from ..common.config import Config
from ..nas.arch_trainer import ArchTrainer
from ..common import utils
from ..nas.model import Model
from ..common.metrics import Metrics
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_model_desc import draw_model_desc


class BilevelArchTrainer(ArchTrainer):
    def __init__(self, conf_train: Config, model: Model, device) -> None:
        super().__init__(conf_train, model, device)

        self._conf_w_optim = conf_train['optimizer']
        self._conf_w_lossfn = conf_train['lossfn']
        self._conf_alpha_optim = conf_train['alpha_optimizer']

    @overrides
    def get_optimizer(self) -> Optimizer:
        # return optim that only operates on w, not alphas
        return utils.get_optimizer(self._conf_w_optim, self.model.weights())

    @overrides
    def pre_fit(self, train_dl: DataLoader, val_dl: Optional[DataLoader],
                optim: Optimizer, sched: _LRScheduler) -> None:

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        assert val_dl is not None
        w_momentum = self._conf_w_optim['momentum']
        w_decay = self._conf_w_optim['decay']
        lossfn = utils.get_lossfn(self._conf_w_lossfn).to(self.device)
        alpha_optim = utils.get_optimizer(
            self._conf_alpha_optim, self.model.alphas())
        self._bilevel_optim = _BilevelOptimizer(w_momentum, w_decay, alpha_optim,
                                                self.model, lossfn)

    @overrides
    def pre_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        super().pre_epoch(train_dl, val_dl)

        # prep val set to train alphas
        self._valid_iter = iter(val_dl)  # type: ignore

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        del self._valid_iter # clean up
        super().post_epoch(train_dl, val_dl)

    @overrides
    def pre_step(self, x: Tensor, y: Tensor, optim: Optimizer) -> None:
        super().pre_step(x, y, optim)

        # reset val loader if we exausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)
        x_val, y_val = x_val.to(self.device), y_val.to(
            self.device, non_blocking=True)

        # update alphas
        self._bilevel_optim.step(x, y, x_val, y_val, optim, self.get_cur_lr())

class _BilevelOptimizer:
    def __init__(self, w_momentum: float, w_decay: float, alpha_optim: Optimizer,
                 model: Union[nn.DataParallel, Model], lossfn: _Loss) -> None:
        self._w_momentum = w_momentum  # momentum for w
        self._w_weight_decay = w_decay  # weight decay for w
        self._lossfn = lossfn
        self._model = model  # main model with respect to w and alpha

        # create a copy of model which we will use
        # to compute grads for alphas without disturbing
        # original weights
        self._vmodel = copy.deepcopy(model)

        # this is the optimizer to optimize alphas parameter
        self._alpha_optim = alpha_optim

    @staticmethod
    def _get_loss(model, lossfn, x, y):
        logits, *_ = model(x) # might also return aux tower logits
        return lossfn(logits, y)

    def _update_vmodel(self, x, y, lr: float, w_optim: Optimizer) -> None:
        """ Update vmodel with w' (main model has w) """

        # TODO: should this loss be stored for later use?
        loss = _BilevelOptimizer._get_loss(self._model, self._lossfn, x, y)
        gradients = autograd.grad(loss, self._model.weights())

        """update weights in vmodel so we leave main model undisturbed
        The main technical difficulty computing w' without affecting alphas is
        that you can't simply do backward() and step() on loss because loss
        tracks alphas as well as w. So, we compute gradients using autograd and
        do manual sgd update."""
        # TODO: other alternative may be to (1) copy model
        #   (2) set require_grads = False on alphas
        #   (3) loss and step on vmodel (4) set back require_grades = True
        with torch.no_grad():  # no need to track gradient for these operations
            for w, vw, g in zip(
                    self._model.weights(), self._vmodel.weights(), gradients):
                # simulate mometum update on model but put this update in vmodel
                m = w_optim.state[w].get(
                    'momentum_buffer', 0.)*self._w_momentum
                vw.copy_(w - lr * (m + g + self._w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self._model.alphas(), self._vmodel.alphas()):
                va.copy_(a)

    def step(self, x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor,
             w_optim: Optimizer, lr:float) -> None:
        self._alpha_optim.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        self._backward_bilevel(x_train, y_train, x_valid, y_valid,
                               lr, w_optim)

        # at this point we should have model with updated grades for w and alpha
        self._alpha_optim.step()

    def _backward_bilevel(self, x_train, y_train, x_valid, y_valid, lr, w_optim):
        """ Compute unrolled loss and backward its gradients """

        # update vmodel with w', but leave alphas as-is
        # w' = w - lr * grad
        self._update_vmodel(x_train, y_train, lr, w_optim)

        # compute loss on validation set for model with w'
        # wrt alphas. The autograd.grad is used instead of backward()
        # to avoid having to loop through params
        vloss = _BilevelOptimizer._get_loss(
            self._vmodel, self._lossfn, x_valid, y_valid)

        v_alphas = tuple(self._vmodel.alphas())
        v_weights = tuple(self._vmodel.weights())
        v_grads = autograd.grad(vloss, v_alphas + v_weights)

        # grad(L(w', a), a), part of Eq. 6
        dalpha = v_grads[:len(v_alphas)]
        # get grades for w' params which we will use it to compute w+ and w-
        dw = v_grads[len(v_alphas):]

        hessian = self._hessian_vector_product(dw, x_train, y_train)

        # dalpha we have is from the unrolled model so we need to
        # transfer those grades back to our main model
        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self._model.alphas(), dalpha, hessian):
                alpha.grad = da - lr*h
        # now that model has both w and alpha grads,
        # we can run w_optim.step() to update the param values

    def _hessian_vector_product(self, dw, x, y, epsilon_unit=1e-2):
        """
        Implements equation 8

        dw = dw` {L_val(w`, alpha)}
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha {L_trn(w+, alpha)} -dalpha {L_trn(w-, alpha)})/(2*eps)
        eps = 0.01 / ||dw||
        """

        """scale epsilon with grad magnitude. The dw
        is a multiplier on RHS of eq 8. So this scalling is essential
        in making sure that finite differences approximation is not way off
        Below, we flatten each w, concate all and then take norm"""
        # TODO: is cat along dim 0 correct?
        dw_norm = torch.cat([w.view(-1) for w in dw]).norm()
        epsilon = epsilon_unit / dw_norm

        # w+ = w + epsilon * grad(w')
        with torch.no_grad():
            for p, v in zip(self._model.weights(), dw):
                p += epsilon * v

        # Now that we have model with w+, we need to compute grads wrt alphas
        # This loss needs to be on train set, not validation set
        loss = _BilevelOptimizer._get_loss(self._model, self._lossfn, x, y)
        dalpha_plus = autograd.grad(
            loss, self._model.alphas())  # dalpha{L_trn(w+)}

        # get model with w- and then compute grads wrt alphas
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, v in zip(self._model.weights(), dw):
                # we had already added dw above so sutracting twice gives w-
                p -= 2. * epsilon * v

        # similarly get dalpha_minus
        loss = _BilevelOptimizer._get_loss(self._model, self._lossfn, x, y)
        dalpha_minus = autograd.grad(loss, self._model.alphas())

        # reset back params to original values by adding dw
        with torch.no_grad():
            for p, v in zip(self._model.weights(), dw):
                p += epsilon * v

        # apply eq 8, final difference to compute hessian
        h = [(p - m) / (2. * epsilon)
             for p, m in zip(dalpha_plus, dalpha_minus)]
        return h
