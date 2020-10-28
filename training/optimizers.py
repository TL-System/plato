"""
Optimizers for training workloads.
"""
from torch import optim

from models.base import Model

def get_optimizer(config, model: Model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if config.training.optimizer == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum
        )
    elif config.training.optimizer == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )

    raise ValueError('No such optimizer: {}'.format(config.training.optimizer))
