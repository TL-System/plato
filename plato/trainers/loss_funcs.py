"""
Loss functions for training workloads.
"""
from torch import nn

from plato.config import Config


def get():
    """Get a loss function with its name from the configuration file."""
    registered_loss_functions = {
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "NLLLoss": nn.NLLLoss,
    }

    loss_function_name = (
        Config().trainer.loss_func
        if hasattr(Config.trainer, "loss_func")
        else "CrossEntropyLoss"
    )
    loss_function = registered_loss_functions.get(loss_function_name)

    if hasattr(Config().parameters, "loss_func"):
        loss_function_params = Config().parameters.loss_func._asdict()
    else:
        loss_function_params = {}

    return loss_function(**loss_function_params)
