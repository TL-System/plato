"""
Utility functions for pruning.
"""

import os
from math import floor

import numpy as np
import torch
from torch.nn.utils import prune

from plato.config import Config


def make_init_mask(model):
    """
    Makes the initial pruning mask for the given model. For example,
    for LeNet-5 architecture it return a list of 5 arrays, each array
    is the same size of each layer's weights and with all 1 entries.
    We do not prune bias
    :param model: a pytorch model
    :return mask: a list of pruning masks
    """
    mask = []
    for __, layer in model.named_parameters():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            tensor = layer.weight.detach().cpu().numpy()
            mask.append(np.ones_like(tensor))

    return mask


def compute_pruned_amount(model, client_id):
    """
    Computes the pruned percentage
    :param model: a pytorch model
    :return pruned percentage, number of remaining weights:
    """
    model_name = Config().trainer.model_name
    checkpoint_path = Config().params["checkpoint_path"]

    mask_path = f"{checkpoint_path}/{model_name}_client{client_id}_mask.pth"

    if not os.path.exists(mask_path):
        return 0

    nonzero = 0
    total = 0
    for _, parameter in model.named_parameters():
        tensor = parameter.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params

    return 100 * (total - nonzero) / total


def structured_pruning(model, pruning_rate, adjust_rate=0.0):
    """
    Conducts structured pruning of a model layer by layer
    model: pytorch model
    pruning_rate: the percentage of all parameters that should be pruned
    adjust_rate: a parameter indicating whether the inputted
    pruning_rate needs to be adjusted for each individual layer to meet
    the total model pruning amount
    """
    norm = 1
    dim = 0
    pruning_rates = []
    mask = []

    if adjust_rate == 0:
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                pruning_rates.append(pruning_rate)
    else:
        total_params = 0
        total_weight_params = 0
        weight_nums = []

        for __, param in model.named_parameters():
            total_params += param.numel()
        total_prune = floor(pruning_rate * total_params)

        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                weight_nums.append(layer.weight.numel())
                pruning_rates.append(0)
        total_weight_params = sum(weight_nums)

        for step, __ in enumerate(weight_nums):
            pruning_rates[step] = (
                (weight_nums[step] / total_weight_params) * total_prune
            ) / weight_nums[step]
            pruning_rates[step] = (pruning_rates[step] * 100) / (100 - adjust_rate)

    step = 0
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            amount = pruning_rates[step]
            prune.ln_structured(layer, "weight", amount, norm, dim)
            for name, buffer in layer.named_buffers():
                if "mask" in name:
                    mask.append(buffer.cpu().numpy())
            step += 1
            prune.remove(layer, "weight")

    return mask


def apply_mask(model, mask, device):
    """Apply the mask onto the model."""

    if not torch.is_tensor(mask[0]):
        for step, __ in enumerate(mask):
            mask[step] = torch.from_numpy(mask[step])

    step = 0
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            device = layer.weight.device
            prune.custom_from_mask(layer, "weight", mask[step].to(device))
            step += 1
    return model
