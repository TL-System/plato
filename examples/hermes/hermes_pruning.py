"""
Utility functions for pruning.
"""

import copy
from collections import OrderedDict
from math import floor
import os
import numpy as np
import torch
from torch.nn.utils import prune
from plato.config import Config


def make_init_mask(model):
    """
    Make the initial pruning mask for the given model. For example,
    for LeNet-5 architecture it return a list of 5 arrays, each array
    is the same size of each layer's weights and with all 1 entries.
    We do not prune bias
    :param model: a pytorch model
    :return mask: a list of pruning masks
    """
    mask = OrderedDict()
    for name, layer in model.state_dict().items():
        tensor = layer.detach().cpu().numpy()
        mask[name] = np.ones_like(tensor)

    return mask


def compute_pruned_amount(model, client_id):
    """
    Compute the pruned percentage
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
    Conduct structured pruning of a model layer by layer
    model: pytorch model
    pruning_rate: the percentage of all parameters that should be pruned
    adjust_rate: a parameter indicating whether the inputted
    pruning_rate needs to be adjusted for each individual layer to meet
    the total model pruning amount
    """
    norm = 1
    dim = 0
    pruning_rates = []

    # The binary masks for all layers
    mask = OrderedDict()

    if adjust_rate == 0:
        for __, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, torch.nn.Linear
            ):
                pruning_rates.append(pruning_rate)
    else:
        total_params = 0
        total_weight_params = 0
        weight_nums = []

        for __, param in model.named_parameters():
            total_params += param.numel()
        total_prune = floor(pruning_rate * total_params)

        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, torch.nn.Linear
            ):
                weight_nums.append(module.weight.numel())
                pruning_rates.append(0)
        total_weight_params = sum(weight_nums)

        for step, __ in enumerate(weight_nums):
            pruning_rates[step] = (
                (weight_nums[step] / total_weight_params) * total_prune
            ) / weight_nums[step]
            pruning_rates[step] = (pruning_rates[step] * 100) / (100 - adjust_rate)

        step = 0

    step = 0
    for __, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            amount = pruning_rates[step]
            prune.ln_structured(module, "weight", amount, norm, dim)
            step += 1

    # Create the mask to be sent to the server
    for name, layer in model.state_dict().items():
        # Any binary masks will have names ending in "_mask" while their masked layers
        # will have names ending with "_orig" according to the pytorch documentation
        # for pruning
        if name.endswith("_mask"):
            mask[name[0 : name.rindex("_mask")]] = layer.cpu().numpy()
        elif not name.endswith("_orig"):
            tensor = layer.detach().cpu().numpy()
            mask[name] = np.ones_like(tensor)

    return mask


def remove(model):
    """
    Remove the original unpruned weight tensors in the model
    """
    for __, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, "weight")


def apply_mask(model, mask, device, pruned_layer_names):
    """Apply the mask onto the model."""

    masked_model = copy.deepcopy(model).to(device)
    if not torch.is_tensor(mask[0]):
        for step, __ in enumerate(mask):
            mask[step] = torch.from_numpy(mask[step])

    step = 0
    for module in masked_model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            device = module.weight.device
            prune.custom_from_mask(
                module, "weight", mask[pruned_layer_names[step]].to(device)
            )
            step += 1

    return masked_model
