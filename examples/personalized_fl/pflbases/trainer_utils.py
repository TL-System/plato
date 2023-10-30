"""
The necessary tools used by trainers.
"""
import os
import re
import random
import logging
from typing import Optional, List, Tuple

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


def search_latest_checkpoint_file(
    checkpoints_dir: str,
    search_key_words: List[str],
    anchor_metric: str = "round",
    filter_words: Optional[List[str]] = None,
):
    """Search the latest checkpoint file under the checkpoint dir based on 'search_key_words'.
    The 'anchor_metric' is utilized to measure what is the latest.
    The 'filter_words' term is utilized to filter out unrelated files.
    """

    if filter_words is None:
        filter_words = ["epochs"]

    def is_filterd_file(ckp_file):
        return any(word in ckp_file for word in filter_words)

    def is_required_file(ckp_file):
        return all(word in ckp_file for word in search_key_words if word is not None)

    checkpoint_files = [
        ckp_file
        for ckp_file in os.listdir(checkpoints_dir)
        if not is_filterd_file(ckp_file)
        and is_required_file(ckp_file)
        and re.search(r"\.pth$", ckp_file)
    ]

    latest_checkpoint_filename = None
    latest_number = 0
    for ckp_file in checkpoint_files:
        pattern = re.escape(anchor_metric) + r"(\d+)"
        obtained_anchor = re.search(pattern, ckp_file, re.IGNORECASE)
        anchor_value = int(obtained_anchor.group(1))
        if anchor_value >= latest_number:
            latest_number = anchor_value
            latest_checkpoint_filename = ckp_file

    return latest_checkpoint_filename


def is_vaild_checkpoint_file(checkpoints_dir, filename: str):
    """Check whether the file exists."""
    file_path = os.path.join(checkpoints_dir, filename)
    if os.path.exists(file_path):
        return True
    return False


def search_checkpoint_file(
    checkpoints_dir: str,
    filename: str,
    key_words: List[str],
    anchor_metric: str = "round",
    mask_words: Optional[List[str]] = None,
    use_latest: bool = True,
) -> Tuple[str, bool]:
    # pylint:disable=too-many-arguments

    """Search for the folder for the target checkpoint file.
    Return the latest file when required."""

    if mask_words is None:
        mask_words = ["epoch"]

    if os.path.exists(os.path.join(checkpoints_dir, filename)):
        return filename, True
    else:
        if use_latest:
            # Loading the latest checkpoint file
            # search_key_words = [model_name, prefix]
            filename = search_latest_checkpoint_file(
                checkpoints_dir,
                search_key_words=key_words,
                anchor_metric=anchor_metric,
                filter_words=mask_words,
            )
            return filename, filename is not None

    return filename, False
