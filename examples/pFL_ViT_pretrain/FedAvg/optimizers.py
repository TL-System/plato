"""
Optimizers for training workloads.

"""

import torch_optimizer as torch_optim
from torch import optim

from plato.config import Config


def get_target_optimizer(stage_prefix: None):
    """ Get an optimizer based on the required prefix that defines the specific stage,
        such as personalized learning stage. """

    optimizers_prefix = {None: "optimizer", "pers": "pers_optimizer"}

    config_optimizer = optimizers_prefix[stage_prefix]

    optimizer_name = getattr(Config().trainer, config_optimizer)
    optim_params = getattr(Config().parameters, config_optimizer)._asdict()
    return optimizer_name, optim_params


def get(model, stage_prefix=None) -> optim.Optimizer:
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

    optimizer_name, optim_params = get_target_optimizer(stage_prefix)

    optimizer = registered_optimizers.get(optimizer_name)
    if optimizer is not None:
        return optimizer(model.parameters(), **optim_params)

    raise ValueError(f"No such optimizer: {optimizer_name}")
