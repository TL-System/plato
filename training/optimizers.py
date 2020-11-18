"""
Optimizers for training workloads.
"""

import bisect
from torch import optim
import numpy as np

from models.base import Model
from config import Config
from utils.step import Step


def get_optimizer(model: Model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if Config().training.optimizer == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=Config().training.learning_rate,
            momentum=Config().training.momentum,
            weight_decay=Config().training.weight_decay
        )
    elif Config().training.optimizer == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=Config().training.learning_rate,
            weight_decay=Config().training.weight_decay
        )

    raise ValueError('No such optimizer: {}'.format(Config().training.optimizer))



def get_lr_schedule(optimizer: optim.Optimizer, iterations_per_epoch: int):
    lambdas = [lambda it: 1.0]

    # Drop the learning rate according to gamma at the specified milestones
    if Config().training.lr_gamma == 0.0:
        raise ValueError('Gamma must be set to produce the learning rate schedule.')

    if Config().training.lr_milestone_steps:
        milestones = [Step.from_str(x, iterations_per_epoch).iteration
                      for x in Config().training.lr_milestone_steps.split(',')]
        lambdas.append(lambda it: Config().training.lr_gamma ** bisect.bisect(milestones, it))

    # Add linear learning rate warmup if specified
    if Config().training.lr_warmup_steps:
        warmup_iters = Step.from_str(Config().training.lr_warmup_steps, iterations_per_epoch).iteration
        lambdas.append(lambda it: min(1.0, it / warmup_iters))

    # Combine the lambdas
    return optim.lr_scheduler.LambdaLR(optimizer, lambda it: np.product([l(it) for l in lambdas]))
