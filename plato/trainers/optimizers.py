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
                    lr=Config().trainer.learning_rate,
                    momentum=Config().trainer.momentum,
                    weight_decay=Config().trainer.weight_decay,
                ),
            ),
            (
                "Adam",
                optim.Adam(
                    model.parameters(),
                    lr=Config().trainer.learning_rate,
                    weight_decay=Config().trainer.weight_decay,
                ),
            ),
            (
                "Adadelta",
                optim.Adadelta(
                    model.parameters(),
                    lr=Config().trainer.learning_rate,
                    rho=Config().trainer.rho,
                    eps=float(Config().trainer.eps),
                    weight_decay=Config().trainer.weight_decay,
                ),
            ),
            (
                "AdaHessian",
                torch_optim.Adahessian(
                    model.parameters(),
                    lr=Config().trainer.learning_rate,
                    betas=(Config().trainer.momentum_b1, Config().trainer.momentum_b2),
                    eps=float(Config().trainer.eps),
                    weight_decay=Config().trainer.weight_decay,
                    hessian_power=Config().trainer.hessian_power,
                ),
            ),
        ]
    )
    optimizer_name = Config().trainer.optimizer

    optimizer = None

    optimizer = registered_optimizers.get(optimizer_name, None)

    if optimizer is None:
        raise ValueError(f"No such optimizer: {optimizer_name}")

    return optimizer
