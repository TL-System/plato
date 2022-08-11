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


def get_optimizer(model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    registered_optimizers = OrderedDict(
        [
            (
                "SGD",
                optim.SGD(
                    model.parameters(),
                    lr=Config().trainer.learning_rate
                    if hasattr(Config().trainer, "learning_rate")
                    else 0.001,
                    momentum=Config().trainer.momentum
                    if hasattr(Config().trainer, "momentum")
                    else 0.937,
                    weight_decay=Config().trainer.weight_decay
                    if hasattr(Config().trainer, "weight_decay")
                    else 0.00058,
                ),
            ),
            (
                "Adam",
                optim.Adam(
                    model.parameters(),
                    lr=Config().trainer.learning_rate
                    if hasattr(Config().trainer, "learning_rate")
                    else 0.001,
                    weight_decay=Config().trainer.weight_decay
                    if hasattr(Config().trainer, "weight_decay")
                    else 0.00058,
                ),
            ),
            (
                "Adadelta",
                optim.Adadelta(
                    model.parameters(),
                    lr=Config().trainer.learning_rate
                    if hasattr(Config().trainer, "learning_rate")
                    else 0.001,
                    rho=Config().trainer.rho
                    if hasattr(Config().trainer, "rho")
                    else 1.0,
                    eps=float(Config().trainer.eps)
                    if hasattr(Config().trainer, "eps")
                    else 1e-3,
                    weight_decay=Config().trainer.weight_decay
                    if hasattr(Config().trainer, "weight_decay")
                    else 0.00058,
                ),
            ),
            (
                "AdaHessian",
                torch_optim.Adahessian(
                    model.parameters(),
                    lr=Config().trainer.learning_rate
                    if hasattr(Config().trainer, "learning_rate")
                    else 0.001,
                    betas=(Config().trainer.momentum_b1, Config().trainer.momentum_b2)
                    if hasattr(Config().trainer, "momentum_b1")
                    and hasattr(Config().trainer, "momentum_b2")
                    else (0.9, 0.999),
                    eps=float(Config().trainer.eps)
                    if hasattr(Config().trainer, "eps")
                    else 1e-3,
                    weight_decay=Config().trainer.weight_decay
                    if hasattr(Config().trainer, "weight_decay")
                    else 0.00058,
                    hessian_power=Config().trainer.hessian_power
                    if hasattr(Config().trainer, "hessian_power")
                    else 1.0,
                ),
            ),
        ]
    )
    optimizer_name = Config().trainer.optimizer

    optimizer = None

    optimizer = registered_optimizers.get(optimizer_name, None)

    print("IN HERE WORKING!", optimizer)

    if optimizer is None:
        raise ValueError(f"No such optimizer: {optimizer_name}")

    return optimizer
