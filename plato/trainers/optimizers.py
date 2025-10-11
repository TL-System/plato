"""
Optimizers for training workloads.
"""

from typing import Union

import torch_optimizer as torch_optim
from timm import optim as timm_optim
from torch import optim

from plato.config import Config


def get(model, **kwargs: Union[str, dict]) -> optim.Optimizer:
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
        "LARS": timm_optim.lars.Lars,
    }

    optimizer_name = (
        kwargs["optimizer_name"]
        if "optimizer_name" in kwargs
        else Config().trainer.optimizer
    )
    optimizer_params = (
        kwargs["optimizer_params"]
        if "optimizer_params" in kwargs
        else Config().parameters.optimizer._asdict()
    )

    # Ensure eps is a float
    if "eps" in optimizer_params:
        optimizer_params["eps"] = float(optimizer_params["eps"])

    optimizer = registered_optimizers.get(optimizer_name)
    if optimizer is not None:
        return optimizer(model.parameters(), **optimizer_params)

    raise ValueError(f"No such optimizer: {optimizer_name}")
