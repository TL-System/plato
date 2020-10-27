from torch import optim

from models.base import Model

# Training settings
lr = 0.01
momentum = 0.5
log_interval = 10

def get_optimizer(optimizer_name, model: Model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if optimizer_name == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum
        )
    elif optimizer_name == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=lr
        )

    raise ValueError('No such optimizer: {}'.format(optimizer_name))
