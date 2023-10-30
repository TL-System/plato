"""
The necessary tools used by trainers.
"""
import random
import logging

import torch
import numpy as np

from pflbases import fedavg_partial


def set_random_seeds(seed: int = 0):
    """Setting the random seed for all parts toward reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def freeze_model(model, modules_name=None, log_info: str = ""):
    """Freezing a part of the model."""
    if modules_name is not None:
        frozen_params = []
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in modules_name):
                param.requires_grad = False
                frozen_params.append(name)

        if log_info is not None:
            logging.info(
                "%s has frozen %s",
                log_info,
                fedavg_partial.Algorithm.extract_modules_name(frozen_params),
            )


def activate_model(model, modules_name=None):
    """Defreezing a part of the model."""
    if modules_name is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in modules_name):
                param.requires_grad = True


def weights_reinitialize(module: torch.nn.Module):
    """Reinitialize a model with the desired seed."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def compute_model_statistics(model):
    """Getting the statistics of the model."""
    weight_statistics = {"mean": 0.0, "variance": 0.0, "std_dev": 0.0}

    total_params = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            values = param.data.cpu().numpy()
            total_params += values.size
            weight_statistics["mean"] += values.mean()
            weight_statistics["variance"] += values.var()

    weight_statistics["mean"] /= total_params
    weight_statistics["variance"] /= total_params
    weight_statistics["std_dev"] = weight_statistics["variance"] ** 0.5

    return weight_statistics
