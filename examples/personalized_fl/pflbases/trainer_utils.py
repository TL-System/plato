"""
The necessary tools used by trainers.
"""
import os
import re


def freeze_model(model, module_names=None):
    """Freezing a part of the model."""
    if module_names is not None:
        frozen_params = []
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in module_names):
                param.requires_grad = False
                frozen_params.append(name)


def activate_model(model, module_names=None):
    """Defreezing a part of the model."""
    if module_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in module_names):
                param.requires_grad = True


def get_latest_checkpoint(checkpoint_dir: str):
    """Get the latest checkpoint file from the given directory."""
    checkpoint_files = [
        ckp_file
        for ckp_file in os.listdir(checkpoint_dir)
        if re.search(r"\.pth$", ckp_file)
    ]

    latest_checkpoint_filename = None
    latest_number = 0
    for ckp_file in checkpoint_files:
        pattern = r"round(\d+)"
        obtained_round = re.search(pattern, ckp_file, re.IGNORECASE)
        round_number = int(obtained_round.group(1))
        if round_number >= latest_number:
            latest_number = round_number
            latest_checkpoint_filename = ckp_file

    return latest_checkpoint_filename
