from atexit import register
import bisect
import sys
from collections import OrderedDict

import numpy as np
from torch import optim
from torch import nn
import torch_optimizer as torch_optim

from plato.config import Config
from plato.utils.step import Step

registered_loss_functions = OrderedDict(
    [
        ("CrossEntropyLoss", nn.CrossEntropyLoss()),
        #("MSELoss", nn.MSELoss()),
        #("MAELoss", nn.L1Loss()),
        ("NLLLoss", nn.NLLLoss()),
        #("BCELoss", nn.BCELoss()),
    ]
)


def get():
    loss_function_name = (
        Config().trainer.loss_func
        if hasattr(Config().trainer, "loss_func")
        else "CrossEntropyLoss"
    )

    loss_function = None

    loss_function = registered_loss_functions.get(loss_function_name, None)

    if loss_function is None:
        raise ValueError(f"No such loss function: {loss_function_name}")

    return loss_function
