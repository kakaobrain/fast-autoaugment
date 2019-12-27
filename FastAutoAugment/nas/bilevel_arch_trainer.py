from typing import Optional, Tuple, Union
import os
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from overrides import overrides

from ..common.config import Config
from ..nas.arch_trainer import ArchTrainer, ArchOptimizer
from ..common.optimizer import get_lr_scheduler, get_optimizer, get_lossfn
from ..nas.model import Model
from ..common.metrics import Metrics
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_model_desc import draw_model_desc


class BilevelArchTrainer(ArchTrainer):
    def __init__(self, conf_common:Config, conf_search:Config,
                 model:Model, device)-> None:
        # region conf vars
        # common
        self.logger_freq = conf_common['logger_freq']
        self.plotsdir = conf_common['plotsdir']
        # search
        conf_lossfn   = conf_search['lossfn']
        self.max_final_edges = conf_search['max_final_edges']
        # optimizers
        self.conf_w_opt    = conf_search['weights']['optimizer']
        self.conf_a_opt    = conf_search['alphas']['optimizer']
        self.w_momentum    = self.conf_w_opt['momentum']
        self.w_decay       = self.conf_w_opt['decay']
        self.grad_clip     = self.conf_w_opt['clip']
        self.conf_w_sched  = conf_search['weights']['lr_schedule']
        # endregion

        self.model = model
        self.device = device
        self.lossfn = get_lossfn(conf_lossfn).to(device)

    @overrides
    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader], epochs:int)\
            ->Tuple[ModelDesc, Metrics, Optional[Metrics]]:
        # optimizer for w and alphas
        w_optim = get_optimizer(self.conf_w_opt, self.model.weights())
        alpha_optim = get_optimizer(self.conf_a_opt, self.model.alphas())
        lr_scheduler = get_lr_scheduler(self.conf_w_sched, epochs, w_optim)

        bilevel_optim = BilevelOptimizer(self.w_momentum, self.w_decay, alpha_optim,
                                     self.model, self.lossfn)

        trainer = BilevelTrainer(bilevel_optim,
            self.max_final_edges, self.plotsdir, self.model, self.device,
            self.lossfn, self.lossfn,
            aux_weight=0.0, grad_clip=self.grad_clip, drop_path_prob=0.0,
            logger_freq=self.logger_freq, title='search_train',
            val_logger_freq=1000, val_title='search_val')

        train_metrics, val_metrics = trainer.fit(train_dl, val_dl, epochs,
                                                 w_optim, lr_scheduler)

        val_metrics.report_best() if val_metrics else train_metrics.report_best()

        return trainer.best_model_desc, train_metrics, val_metrics

class BilevelTrainer(Trainer):
    def __init__(self, arch_optim,
                 max_final_edges:int, plotsdir:str,
                 model:Model, device, train_lossfn: _Loss, test_lossfn: _Loss,
                 aux_weight: float, grad_clip: float,
                 drop_path_prob: float, logger_freq: int, title:str,
                val_logger_freq:int, val_title:str)->None:
        super().__init__(model, device, train_lossfn, test_lossfn,
                         aux_weight=aux_weight, grad_clip=grad_clip,
                         drop_path_prob=drop_path_prob,
                         logger_freq=logger_freq, title=title,
                         val_logger_freq=val_logger_freq, val_title=val_title)

        self._arch_optim = arch_optim
        self.max_final_edges, self.plotsdir = max_final_edges, plotsdir
        self.best_model_desc = self._get_model_desc()

    @overrides
    def pre_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        super().pre_epoch(train_dl, val_dl, train_metrics,  val_metrics)

        # prep val set to train alphas
        assert val_dl is not None
        self._valid_iter = iter(val_dl)

    @overrides
    def pre_step(self, x:Tensor, y:Tensor, optim:Optimizer, train_metrics:Metrics)->None:
        super().pre_step(x, y, optim, train_metrics)

        # reset val loader if we exausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device, non_blocking=True)

        # update alphas
        self._arch_optim.step(x, y, x_val, y_val, optim, train_metrics)

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader],
                  train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        del self._valid_iter # clean up
        super().post_epoch(train_dl, val_dl, train_metrics, val_metrics)

        self._update_best(train_metrics, val_metrics)

    def _update_best(self, train_metrics:Metrics, val_metrics:Optional[Metrics])->None:
        if (val_metrics and val_metrics.is_best()) or \
                (not val_metrics and train_metrics.is_best()):
            self.best_model_desc = self._get_model_desc()

            # log model_desc as a image
            plot_filepath = os.path.join(self.plotsdir, "EP{train_metrics.epoch:03d}")
            draw_model_desc(self.best_model_desc, plot_filepath+"-normal",
                            caption=f"Epoch {train_metrics.epoch}")

    def _get_model_desc(self)->ModelDesc:
        return self.model.finalize(max_edges=self.max_final_edges)


def _get_loss(model, lossfn, x, y):
    logits, *_ = model(x)
    return lossfn(logits, y)


class BilevelOptimizer(ArchOptimizer):
    def __init__(self, w_momentum:float, w_decay:float, alpha_optim:Optimizer,
                 model:Union[nn.DataParallel, Model], lossfn:_Loss)->None:
        self._w_momentum = w_momentum # momentum for w
        self._w_weight_decay = w_decay # weight decay for w
        self._lossfn = lossfn
        self._model = model # main model with respect to w and alpha

        # create a copy of model which we will use
        # to compute grads for alphas without disturbing
        # original weights
        # TODO: see if there are any issues in deepcopy for pytorch
        self._vmodel = copy.deepcopy(model)

        # this is the optimizer to optimize alphas parameter
        self._alpha_optim = alpha_optim

    def _update_vmodel(self, x, y, lr:float, w_optim:Optimizer)->None:
        """ Update vmodel with w' (main model has w) """

        # TODO: should this loss be stored for later use?
        loss = _get_loss(self._model, self._lossfn, x, y)
        gradients = autograd.grad(loss, self._model.weights())

        """update weights in vmodel so we leave main model undisturbed
        The main technical difficulty computing w' without affecting alphas is
        that you can't simply do backward() and step() on loss because loss
        tracks alphas as well as w. So, we compute gradients using autograd and
        do manual sgd update."""
        # TODO: other alternative may be to (1) copy model
        #   (2) set require_grads = False on alphas
        #   (3) loss and step on vmodel (4) set back require_grades = True
        with torch.no_grad(): # no need to track gradient for these operations
            for w, vw, g in zip(
                    self._model.weights(), self._vmodel.weights(), gradients):
                # simulate mometum update on model but put this update in vmodel
                m = w_optim.state[w].get('momentum_buffer', 0.)*self._w_momentum
                vw.copy_(w - lr * (m + g + self._w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self._model.alphas(), self._vmodel.alphas()):
                va.copy_(a)

    @overrides
    def step(self, x_train:Tensor, y_train:Tensor, x_valid:Tensor, y_valid:Tensor,
            w_optim:Optimizer, train_metrics:Metrics)->None:
        self._alpha_optim.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        self._backward_bilevel(x_train, y_train, x_valid, y_valid,
                               train_metrics.get_cur_lr(), w_optim)

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
        vloss = _get_loss(self._vmodel, self._lossfn, x_valid, y_valid)

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
        dw_norm = torch.cat([w.view(-1) for w in dw]).norm()
        epsilon = epsilon_unit / dw_norm

        # w+ = w + epsilon * grad(w')
        with torch.no_grad():
            for p, v in zip(self._model.weights(), dw):
                p += epsilon * v

        # Now that we have model with w+, we need to compute grads wrt alphas
        # This loss needs to be on train set, not validation set
        loss = _get_loss(self._model, self._lossfn, x, y)
        dalpha_plus = autograd.grad(loss, self._model.alphas()) #dalpha{L_trn(w+)}

        # get model with w- and then compute grads wrt alphas
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, v in zip(self._model.weights(), dw):
                # we had already added dw above so sutracting twice gives w-
                p -= 2. * epsilon * v

        # similarly get dalpha_minus
        loss = _get_loss(self._model, self._lossfn, x, y)
        dalpha_minus = autograd.grad(loss, self._model.alphas())

        # reset back params to original values by adding dw
        with torch.no_grad():
            for p, v in zip(self._model.weights(), dw):
                p += epsilon * v

        # apply eq 8, final difference to compute hessian
        h= [(p - m) / (2. * epsilon) for p, m in zip(dalpha_plus, dalpha_minus)]
        return h

