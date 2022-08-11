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
                    ("lr", Config().trainer.learning_rate),
                    ("momentum", Config().trainer.momentum),
                    ("weight_decay", Config().trainer.weight_decay),
                ]
            ),
        ),
        (
            "Adam",
            OrderedDict(
                [
                    ("lr", Config().trainer.learning_rate),
                    ("weight_decay", Config().trainer.weight_decay),
                ]
            ),
        ),
        (
            "Adadelta",
            OrderedDict(
                [
                    ("lr", Config().trainer.learning_rate),
                    ("rho", Config().trainer.rho),
                    ("eps", float(Config().trainer.eps)),
                    ("weight_decay", Config().trainer.weight_decay),
                ]
            ),
        ),
        (
            "AdaHessian",
            OrderedDict(
                [
                    ("lr", Config().trainer.learning_rate),
                    (
                        "betas",
                        (Config().trainer.momentum_b1, Config().trainer.momentum_b2),
                    ),
                    ("eps", float(Config().trainer.eps)),
                    ("weight_decay", Config().trainer.weight_decay),
                    ("hessian_power", Config().trainer.hessian_power),
                ]
            ),
        ),
    ]
)


def get_optimizer(model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    optimizer_name = Config().trainer.optimizer

    optimizer = registered_optimizers[optimizer_name]
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
    """for name, registered_optimizer in registered_optimizers.items():
        if name.startswith(optimizer_name):
            optimizer = registered_optimizer
        # We have found our optimizer
        if optimizer is not None:
            break"""
