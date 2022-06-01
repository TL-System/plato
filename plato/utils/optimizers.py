"""
Optimizers for training workloads.
"""

import bisect
import sys

import numpy as np
from torch import optim
from torch import nn
import torch_optimizer as torch_optim

from plato.config import Config
from plato.utils.step import Step


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_optimizer(model,
                  optimizer_name=None,
                  learning_rate=None,
                  momentum=None,
                  weight_decay=None) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if optimizer_name is None:
        optimizer_name = Config().trainer.optimizer
    if learning_rate is None:
        learning_rate = Config().trainer.learning_rate
    if weight_decay is None:
        weight_decay = Config().trainer.weight_decay

    if optimizer_name == 'SGD':
        if momentum is None:
            momentum = Config().trainer.momentum
        return optim.SGD(model.parameters(),
                         lr=learning_rate,
                         momentum=momentum,
                         weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':

        return optim.Adadelta(model.parameters(),
                              lr=learning_rate,
                              rho=Config().trainer.rho,
                              eps=float(Config().trainer.eps),
                              weight_decay=weight_decay)
    elif optimizer_name == 'AdaHessian':
        return torch_optim.Adahessian(
            model.parameters(),
            lr=learning_rate,
            betas=(Config().trainer.momentum_b1, Config().trainer.momentum_b2),
            eps=float(Config().trainer.eps),
            weight_decay=weight_decay,
            hessian_power=Config().trainer.hessian_power,
        )

    raise ValueError(f'No such optimizer: {Config().trainer.optimizer}')


def get_lr_schedule(optimizer: optim.Optimizer,
                    iterations_per_epoch: int,
                    lr_schedule=None,
                    epochs=None,
                    train_loader=None):
    """Returns a learning rate scheduler according to the configuration."""
    if lr_schedule is None:
        lr_schedule = Config().trainer.lr_schedule
    if lr_schedule == 'CosineAnnealingLR':
        if epochs is None:
            epochs = Config().trainer.epochs
        return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    len(train_loader) * epochs)
    elif lr_schedule == 'LambdaLR':
        lambdas = [lambda it: 1.0]

        if hasattr(Config().trainer, 'lr_gamma') and hasattr(
                Config().trainer, 'lr_milestone_steps'):
            milestones = [
                Step.from_str(x, iterations_per_epoch).iteration
                for x in Config().trainer.lr_milestone_steps.split(',')
            ]
            lambdas.append(lambda it: Config().trainer.lr_gamma**bisect.bisect(
                milestones, it))

        # Add a linear learning rate warmup if specified
        if hasattr(Config().trainer, 'lr_warmup_steps'):
            warmup_iters = Step.from_str(Config().trainer.lr_warmup_steps,
                                         iterations_per_epoch).iteration
            lambdas.append(lambda it: min(1.0, it / warmup_iters))

        # Combine the lambdas
        return optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: np.product([l(it) for l in lambdas]))
    elif lr_schedule == 'StepLR':
        step_size = Config().trainer.lr_step_size if hasattr(
            Config().trainer, 'lr_step_size') else 30
        gamma = Config().trainer.lr_gamma if hasattr(Config().trainer,
                                                     'lr_gamma') else 0.1
        return optim.lr_scheduler.StepLR(optimizer,
                                         step_size=step_size,
                                         gamma=gamma)
    elif lr_schedule == 'ReduceLROnPlateau':
        factor = Config().trainer.lr_factor if hasattr(Config().trainer,
                                                       'lr_factor') else 0.1
        patience = Config().trainer.lr_patience if hasattr(
            Config().trainer, 'lr_patience') else 10
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    factor=factor,
                                                    patience=patience)
    else:
        sys.exit('Error: Unknown learning rate scheduler.')


def get_loss_criterion():
    """Obtain the loss criterion used for training the model."""
    if Config().trainer.loss_criterion == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()
