from torch import optim

from models.base import Model

# Training settings
lr = 0.01
momentum = 0.5
log_interval = 10

def get_optimizer(model: Model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
