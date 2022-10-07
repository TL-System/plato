"""
Optimizers for training workloads.
"""

import torch_optimizer as torch_optim
from torch import optim

from plato.config import Config


def get(model) -> optim.Optimizer:
    """Get an optimizer with its name and parameters obtained from the configuration file."""
    registered_optimizers = {
        "Adam": optim.Adam,
        "Adadelta": optim.Adadelta,
        "Adagrad": optim.Adagrad,
        "AdaHessian": torch_optim.Adahessian,
        "AdamW": optim.AdamW,
        "SparseAdam": optim.SparseAdam,
        "Adamax": optim.Adamax,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "NAdam": optim.NAdam,
        "RAdam": optim.RAdam,
        "RMSprop": optim.RMSprop,
        "Rprop": optim.Rprop,
        "SGD": optim.SGD,
    }

    optimizer_name = Config().trainer.optimizer
    optim_params = Config().parameters.optimizer._asdict()
    optimizer = registered_optimizers.get(optimizer_name)
    if optimizer is not None:
        return optimizer(model.parameters(), **optim_params)

    raise ValueError(f"No such optimizer: {optimizer_name}")
