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
from ..nas.arch_trainer import ArchTrainer
from ..common.optimizer import get_lr_scheduler, get_optimizer, get_lossfn
from ..nas.model import Model
from ..common.metrics import Metrics
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_genotype import draw_model_desc


class BilevelArchTrainer(ArchTrainer):
    @overrides
    def fit(self, conf_search:Config, model:Model, device,
            train_dl:DataLoader, val_dl:Optional[DataLoader], epochs:int,
            plotsdir:str, logger_freq:int)->Tuple[ModelDesc, Metrics, Optional[Metrics]]:
        # region conf vars
        # search
        bilevel       = conf_search['bilevel']
        conf_lossfn   = conf_search['lossfn']
        max_final_edges = conf_search['max_final_edges']
        # optimizers
        conf_w_opt    = conf_search['weights']['optimizer']
        w_momentum    = conf_w_opt['momentum']
        w_decay       = conf_w_opt['decay']
        grad_clip     = conf_w_opt['clip']
        conf_w_sched  = conf_search['weights']['lr_schedule']
        conf_a_opt    = conf_search['alphas']['optimizer']
        # endregion

        lossfn = get_lossfn(conf_lossfn).to(device)

        # optimizer for w and alphas
        w_optim = get_optimizer(conf_w_opt, model.weights())
        alpha_optim = get_optimizer(conf_a_opt, model.alphas())
        lr_scheduler = get_lr_scheduler(conf_w_sched, epochs, w_optim)

        trainer = BilevelTrainer(w_momentum, w_decay, alpha_optim,
            max_final_edges, plotsdir, model, device, lossfn, lossfn,
            aux_weight=0.0, grad_clip=grad_clip, drop_path_prob=0.0,
            logger_freq=logger_freq, tb_tag='search_train',
            val_logger_freq=1000, val_tb_tag='search_val')

        train_metrics, val_metrics = trainer.fit(train_dl, val_dl, epochs,
                                                 w_optim, lr_scheduler)

        val_metrics.report_best() if val_metrics else train_metrics.report_best()

        return trainer.best_model_desc, train_metrics, val_metrics

class BilevelTrainer(Trainer):
    def __init__(self, w_momentum:float, w_decay:float, alpha_optim:Optimizer,
                 max_final_edges:int, plotsdir:str,
                 model:Model, device, train_lossfn: _Loss, test_lossfn: _Loss,
                 aux_weight: float, grad_clip: float,
                 drop_path_prob: float, logger_freq: int,
                 tb_tag: str, val_logger_freq:int, val_tb_tag:str)->None:
        super().__init__(model, device, train_lossfn, test_lossfn,
                         aux_weight=aux_weight, grad_clip=grad_clip,
                         drop_path_prob=drop_path_prob,
                         logger_freq=logger_freq, tb_tag=tb_tag)

        self.w_momentum, self.w_decay = w_momentum, w_decay
        self.alpha_optim = alpha_optim
        self.max_final_edges, self.plotsdir = max_final_edges, plotsdir
        self._bilevel_optim = BilevelOptimizer(w_momentum, w_decay, alpha_optim,
                                     True, model, train_lossfn)
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
        self._bilevel_optim.step(x, y, x_val, y_val, train_metrics.get_cur_lr(), optim)

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

# t.view(-1) reshapes tensor to 1 row N columnsstairs
#   w - model parameters
#   alphas - arch parameters
#   w' - updated w using grads from the loss
class BilevelOptimizer:
    def __init__(self, w_momentum:float, w_decay:float, alpha_optim:Optimizer,
                 bilevel:bool, model:Union[nn.DataParallel, Model],
                 lossfn:_Loss)->None:
        self._w_momentum = w_momentum # momentum for w
        self._w_weight_decay = w_decay # weight decay for w
        self._lossfn = lossfn
        self._model = model # main model with respect to w and alpha
        self._bilevel:bool = bilevel

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

    def step(self, x_train, y_train, x_valid, y_valid, lr:float, w_optim:Optimizer):
        self._alpha_optim.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        if self._bilevel:
            self._backward_bilevel(x_train, y_train, x_valid, y_valid, lr, w_optim)
        else:
            # directly optimize alpha on w, instead of w_pi
            self._backward_classic(x_valid, y_valid)

        # at this point we should have model with updated grades for w and alpha
        self._alpha_optim.step()

    def _backward_classic(self, x_valid, y_valid):
        """
        This function is used only for experimentation to see how much better
        is bilevel optimization vs simply doing it naively
        simply train on validate set and backward
        :param x_valid:
        :param y_valid:
        :return:
        """
        loss = _get_loss(self._model, self._lossfn, x_valid, y_valid)
        # both alphas and w require grad but only alphas optimizer will
        # step in current phase.
        loss.backward()

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

