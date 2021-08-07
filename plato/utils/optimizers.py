"""
Optimizers for training workloads.
"""

import bisect
import sys

import numpy as np
from torch import optim
from torch import nn

from plato.config import Config

from plato.utils.fedprox_optimizer import FedProxOptimizer
from plato.utils.step import Step


def get_optimizer(model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if Config().trainer.optimizer == 'SGD':
        return optim.SGD(model.parameters(),
                         lr=Config().trainer.learning_rate,
                         momentum=Config().trainer.momentum,
                         weight_decay=Config().trainer.weight_decay)
    elif Config().trainer.optimizer == 'Adam':
        return optim.Adam(model.parameters(),
                          lr=Config().trainer.learning_rate,
                          weight_decay=Config().trainer.weight_decay)
    elif Config().trainer.optimizer == 'FedProx':
        return FedProxOptimizer(model.parameters(),
                                lr=Config().trainer.learning_rate,
                                momentum=Config().trainer.momentum,
                                weight_decay=Config().trainer.weight_decay)

    raise ValueError('No such optimizer: {}'.format(
        Config().trainer.optimizer))


def get_lr_schedule(optimizer: optim.Optimizer,
                    iterations_per_epoch: int,
                    train_loader=None):
    """Returns a learning rate scheduler according to the configuration."""
    if Config().trainer.lr_schedule == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            len(train_loader) * Config().trainer.epochs)
    elif Config().trainer.lr_schedule == 'LambdaLR':
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
    elif Config().trainer.lr_schedule == 'StepLR':
        step_size = Config().trainer.lr_step_size if hasattr(
            Config().trainer, 'lr_step_size') else 30
        gamma = Config().trainer.lr_gamma if hasattr(Config().trainer,
                                                     'lr_gamma') else 0.1
        return optim.lr_scheduler.StepLR(optimizer,
                                         step_size=step_size,
                                         gamma=gamma)
    elif Config().trainer.lr_schedule == 'ReduceLROnPlateau':
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
