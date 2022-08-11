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

from collections import OrderedDict


registered_optimizers = OrderedDict(
    [
        ("SGD", optim.SGD),
        ("Adam", optim.Adam),
        ("Adadelta", optim.Adadelta),
        ("AdaHessian", torch_optim.Adahessian),
    ]
)
registered_optimizers_parameters = OrderedDict(
    [
        (
            "SGD",
            OrderedDict(
                [
                    (
                        "lr",
                        Config().trainer.learning_rate
                        if hasattr(Config().trainer, "learning_rate")
                        else 0.001,
                    ),
                    (
                        "momentum",
                        Config().trainer.momentum
                        if hasattr(Config().trainer, "momentum")
                        else 0.937,
                    ),
                    (
                        "weight_decay",
                        Config().trainer.weight_decay
                        if hasattr(Config().trainer, "weight_decay")
                        else 0.00058,
                    ),
                ]
            ),
        ),
        (
            "Adam",
            OrderedDict(
                [
                    (
                        "lr",
                        Config().trainer.learning_rate
                        if hasattr(Config().trainer, "learning_rate")
                        else 0.001,
                    ),
                    (
                        "weight_decay",
                        Config().trainer.weight_decay
                        if hasattr(Config().trainer, "weight_decay")
                        else 0.00058,
                    ),
                ]
            ),
        ),
        (
            "Adadelta",
            OrderedDict(
                [
                    (
                        "lr",
                        Config().trainer.learning_rate
                        if hasattr(Config().trainer, "learning_rate")
                        else 0.001,
                    ),
                    (
                        "rho",
                        Config().trainer.rho
                        if hasattr(Config().trainer, "rho")
                        else 1.0,
                    ),
                    (
                        "eps",
                        float(Config().trainer.eps)
                        if hasattr(Config().trainer, "eps")
                        else 1e-3,
                    ),
                    (
                        "weight_decay",
                        Config().trainer.weight_decay
                        if hasattr(Config().trainer, "weight_decay")
                        else 0.00058,
                    ),
                ]
            ),
        ),
        (
            "AdaHessian",
            OrderedDict(
                [
                    (
                        "lr",
                        Config().trainer.learning_rate
                        if hasattr(Config().trainer, "learning_rate")
                        else 0.001,
                    ),
                    (
                        "betas",
                        (Config().trainer.momentum_b1, Config().trainer.momentum_b2)
                        if hasattr(Config().trainer, "momentum_b1")
                        and hasattr(Config().trainer, "momentum_b2")
                        else (0.9, 0.999),
                    ),
                    (
                        "eps",
                        float(Config().trainer.eps)
                        if hasattr(Config().trainer, "eps")
                        else 1e-3,
                    ),
                    (
                        "weight_decay",
                        Config().trainer.weight_decay
                        if hasattr(Config().trainer, "weight_decay")
                        else 0.00058,
                    ),
                    (
                        "hessian_power",
                        Config().trainer.hessian_power
                        if hasattr(Config().trainer, "hessian_power")
                        else 1.0,
                    ),
                ]
            ),
        ),
    ]
)


def get_optimizer(model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    optimizer_name = Config().trainer.optimizer

    optimizer = registered_optimizers.get(optimizer_name, None)
    if optimizer_name is None:
        raise ValueError(f"No such optimizer: {optimizer_name}")

    optimizer_parameters = registered_optimizers_parameters[optimizer_name]

    if optimizer_name == "SGD":
        optimizer(
            model.parameters(),
            lr=optimizer_parameters["lr"],
            momentum=optimizer_parameters["momentum"],
            weight_decay=optimizer_parameters["weight_decay"],
        )
    elif optimizer_name == "Adam":
        optimizer(
            model.parameters(),
            lr=optimizer_parameters["lr"],
            weight_decay=optimizer_parameters["weight_decay"],
        )
    elif optimizer_name == "Adadelta":
        optimizer(
            model.parameters(),
            lr=optimizer_parameters["lr"],
            rho=optimizer_parameters["rho"],
            eps=optimizer_parameters["eps"],
            weight_decay=optimizer_parameters["weight_decay"],
        )
    elif optimizer_name == "AdaHessian":
        optimizer(
            model.parameters(),
            lr=optimizer_parameters["lr"],
            betas=optimizer_parameters["betas"],
            eps=optimizer_parameters["eps"],
            weight_decay=optimizer_parameters["weight_decay"],
            hessian_power=optimizer_parameters["hessian_power"],
        )

    return optimizer
