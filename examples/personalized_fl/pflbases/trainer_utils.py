"""
The necessary tools used by trainers.
"""
import os
import re
import string
from typing import List


def freeze_model(model, layer_names=None):
    """Freezing a part of the model."""
    if layer_names is not None:
        frozen_params = []
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = False
                frozen_params.append(name)


def activate_model(model, layer_names=None):
    """Defreezing a part of the model."""
    if layer_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
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


def extract_module_names(parameter_names: List[str]):
    """
    Extract module names from the given parameter names. A parameter name is a list of names
    connected by `.`, such as `encoder.conv1.weight`.
    """
    split_char = "."

    # Converting `encoder.conv1.weight`` to [encoder, conv1, weight]
    translator = str.maketrans("", "", string.punctuation)
    splitted_names = [
        [subname.translate(translator).lower() for subname in name.split(split_char)]
        for name in parameter_names
    ]

    # With [encoder, conv1, weight], [encoder, conv1, bias], diff_idx = 1.
    diff_idx = 0
    for idx, subnames in enumerate(zip(*splitted_names)):
        if len(set(subnames)) > 1:
            diff_idx = idx
            break

    # Extract the first `diff_idx` parameter names as module names
    extracted_names = []
    for para_name in parameter_names:
        splitted_names = para_name.split(split_char)
        core_names = splitted_names[: diff_idx + 1]
        module_name = f"{split_char}".join(core_names)
        if module_name not in extracted_names:
            extracted_names.append(module_name)

    return extracted_names
