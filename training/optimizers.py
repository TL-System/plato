"""
Optimizers for training workloads.
"""
from torch import optim

from models.base import Model
from config import Config

def get_optimizer(model: Model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if Config().training.optimizer == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=Config().training.learning_rate,
            momentum=Config().training.momentum
        )
    elif Config().training.optimizer == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=Config().training.learning_rate
        )

    raise ValueError('No such optimizer: {}'.format(Config().training.optimizer))
