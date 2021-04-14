from copy import deepcopy
import logging

import torch
import torch.jit

from torch import nn
from torch.nn import Identity
from torch.optim import Adam, SGD

from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

from conf import cfg


logger = logging.getLogger(__name__)


def collect_bn_params(model, logging_bn_params):
    bn_params, bn_names, bn_names_module = [], set(), set()
    # only optimize the affine parameters after normalization
    for name, _ in model.named_buffers():
        if "running_mean" in name:
            bn_names.add(name.replace('running_mean', 'weight'))
            bn_names.add(name.replace('running_mean', 'bias'))
            bn_names_module.add(name[:-13])  # len(".running_mean") == 13

    for name, params in model.named_parameters():
        if name in bn_names:  # if "bn" in name:
            params.requires_grad = True
            bn_params.append(params)
        else:
            params.requires_grad = False

    if logging_bn_params:
        logger.info('test-time optimized parameters: ')
        logger.info(bn_names)

    return bn_params


def construct_optimizer(optim_param):
    if cfg.OPTIM.METHOD == 'Adam':
        return Adam(optim_param,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return SGD(optim_param,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class SoftmaxEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softmax_entropy(x).mean(0)


def save_model_and_optimizer(model, optimizer):
    """Saves the state dicts of model and optimizer."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Loads the state dicts of model and optimizer."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.enable_grad()
def optim_model(inputs, model, optimizer, loss_fun):
    # Perform the forward pass
    preds = model(inputs)
    # Compute the loss
    loss = loss_fun(preds)
    # Perform the backward pass
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # Update the parameters
    optimizer.step()
    return preds


class tent(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = load_model(model_name, cfg.CKPT_DIR,
                                cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
        self.iter = cfg.OPTIM.ITER
        self.eval_only = cfg.CORRUPTION.EVAL_ONLY
        self.reset_state = cfg.CORRUPTION.RESET_STATE
        self.loss_fun = SoftmaxEntropy().cuda()
        self.optimizer = construct_optimizer(
            collect_bn_params(self.model, False))
        self.model_state, self.optimizer_state = \
            save_model_and_optimizer(self.model, self.optimizer)

    def _reset_state(self, x):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def forward(self, x):
        if self.reset_state:
            self._reset_state(x)

        if not self.eval_only:
            self.model.train()
            for _ in range(self.iter):
                x = optim_model(
                    x, self.model,
                    self.optimizer,
                    self.loss_fun)
        else:
            x = self.model(x)

        return x
