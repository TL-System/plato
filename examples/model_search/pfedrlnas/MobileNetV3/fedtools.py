"""
Helped functions used by trainer and algorithm in PerFedRLNAS.
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from model.mobilenetv3_supernet import NasDynamicModel


def set_active_subnet(model, cfg):
    """Set the suupernet to subnet with given cfg."""
    model.set_active_subnet(
        cfg["resolution"],
        cfg["width"],
        cfg["depth"],
        cfg["kernel_size"],
        cfg["expand_ratio"],
    )


def sample_subnet_wo_config(supernet):
    """Sample a subnet of random structure and return the random structure."""
    subnet_config = supernet.sample_active_subnet()
    subnet = supernet.get_active_subnet(preserve_weight=True)

    return subnet, subnet_config


def sample_subnet_w_config(supernet, cfg, preserver_weight=True):
    """Sample a subnet of with given ViT structure."""
    set_active_subnet(supernet, cfg)
    subnet = supernet.get_active_subnet(preserve_weight=preserver_weight)
    return subnet


def _average_fuse(global_iter, client_iters, num_samples, avg_last=True):
    # pylint: disable=too-many-locals
    weight_numbers = np.zeros(len(client_iters))
    neg_numebers = np.zeros(len(client_iters))
    total_samples = sum(num_samples)

    try:
        while True:
            global_name, global_param = next(global_iter)
            if (not avg_last) and ("classifier" in global_name):
                for client_iter in enumerate(client_iters):
                    _, client_param = next(client_iter)
                continue
            baseline = copy.deepcopy(global_param.data)
            deltas = torch.zeros(baseline.size())
            weight_numbers += 1
            temp_delta = []
            is_update = torch.zeros(deltas.size())
            for idx, client_iter in enumerate(client_iters):
                _, client_param = next(client_iter)
                delta = client_param.data - baseline
                is_update = torch.ones(is_update.size()) - (
                    torch.abs(delta) < 1e-8
                ).type(dtype=is_update.dtype)
                deltas += delta * num_samples[idx] / total_samples * is_update
                temp_delta.append(copy.deepcopy(delta))
            global_param.data += deltas
            for idx, delta in enumerate(temp_delta):
                if (
                    torch.cosine_similarity(
                        torch.flatten(deltas), torch.flatten(delta), dim=0
                    ).item()
                    < 0
                ):
                    neg_numebers[idx] += 1
    except StopIteration:
        pass
    neg_ratio = np.divide(neg_numebers, weight_numbers)
    logging.info("the neg ratio of each client weight is %s", str(neg_ratio))
    return neg_ratio


def fuse_weight(supernet, subnets, cfgs, num_samples):
    """Fuse weights of subnets with different structure into supernet."""
    proxy_supernets = []
    for i, cfg in enumerate(cfgs):
        proxy_supernet = NasDynamicModel()
        subnet = subnets[i]
        proxy_supernet.set_active_subnet(
            cfg["resolution"],
            cfg["width"],
            cfg["depth"],
            cfg["kernel_size"],
            cfg["expand_ratio"],
        )
        proxy_supernet.get_weight_from_subnet(subnet)
        proxy_supernets.append(proxy_supernet)
    proxy_iters = []
    for proxy_supernet in proxy_supernets:
        proxy_iters.append(proxy_supernet.named_parameters())
    global_iter = supernet.named_parameters()
    neg_ratio = _average_fuse(global_iter, proxy_iters, num_samples)
    return neg_ratio


def generate_proxy_supernets(subnets, cfgs):
    """Generate a series of proxy supernets."""
    proxy_supernets = []
    for i, cfg in enumerate(cfgs):
        proxy_supernet = NasDynamicModel()
        subnet = subnets[i]
        proxy_supernet.set_active_subnet(
            cfg["resolution"],
            cfg["width"],
            cfg["depth"],
            cfg["kernel_size"],
            cfg["expand_ratio"],
        )
        proxy_supernet.get_weight_from_subnet(subnet)
        proxy_supernets.append(proxy_supernet)
    return proxy_supernets


def calculate_similarity(model_path, model, update, staleness):
    """Calculate the model similarity"""
    similarity = 1
    if staleness > 1 and os.path.exists(model_path):
        previous_model = copy.deepcopy(model)
        previous_model.load_state_dict(torch.load(model_path))

        previous = torch.zeros(0)
        for __, weight in previous_model.cpu().state_dict().items():
            previous = torch.cat((previous, weight.view(-1)))

        current = torch.zeros(0)
        for __, weight in model.cpu().state_dict().items():
            current = torch.cat((current, weight.view(-1)))

        deltas = torch.zeros(0)
        for __, delta in update.items():
            deltas = torch.cat((deltas, delta.view(-1)))

        similarity = F.cosine_similarity(current - previous, deltas, dim=0)
    return similarity
