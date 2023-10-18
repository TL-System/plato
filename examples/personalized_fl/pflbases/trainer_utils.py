"""
The necessary tools used by trainers.
"""

import logging
from pflbases import fedavg_partial


def freeze_model(model, modules_name=None, log_info: str = ""):
    """Freezing a part of the model."""
    if modules_name is not None:
        frozen_params = []
        for name, param in model.named_parameters():
            if any([param_name in name for param_name in modules_name]):
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
