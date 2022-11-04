from array import array
from NASVIT import models
from plato.config import Config
from NASVIT.misc.config import get_config
from NASVIT.models.attentive_nas_dynamic_model import AttentiveNasDynamicModel
import copy
import logging
import torch
import numpy as np


def sample_subnet_wo_config(supernet):
    subnet_config = supernet.sample_active_subnet()
    subnet = supernet.get_active_subnet(preserve_weight=True)

    return subnet, subnet_config


def sample_subnet_w_config(supernet, cfg, preserver_weight=True):
    supernet.set_active_subnet(
        cfg["resolution"],
        cfg["width"],
        cfg["depth"],
        cfg["kernel_size"],
        cfg["expand_ratio"],
    )
    subnet = supernet.get_active_subnet(preserve_weight=preserver_weight)
    return subnet


def _average_fuse(global_iter, client_iters, num_samples, avg_last=True):
    # total_samples=sum(num_samples)
    weight_numbers = np.zeros(len(client_iters))
    neg_numebers = np.zeros(len(client_iters))
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
            is_updates = torch.zeros(deltas.size())
            for idx, client_iter in enumerate(client_iters):
                _, client_param = next(client_iter)
                delta = client_param.data - baseline
                is_update = torch.where(
                    client_param.data == baseline,
                    torch.zeros(delta.size()),
                    torch.ones(delta.size()),
                )
                deltas += delta * num_samples[idx] * is_update
                temp_delta.append(copy.deepcopy(delta))
                is_updates += is_update * num_samples[idx]
            is_updates = torch.where(
                is_updates == 0, torch.ones(is_updates.size()), is_updates
            )
            # logging.info(is_updates)
            deltas = torch.div(deltas, is_updates)
            for idx, delta in enumerate(temp_delta):
                if (
                    torch.cosine_similarity(
                        torch.flatten(deltas), torch.flatten(delta), dim=0
                    ).item()
                    < 0
                ):
                    neg_numebers[idx] += 1
            global_param.data += deltas
    except StopIteration:
        pass
    neg_ratio = np.divide(neg_numebers, weight_numbers)
    logging.info("the neg ratio of each client weight is %s", str(neg_ratio))
    return neg_ratio


def fuse_weight(supernet, subnets, cfgs, num_samples):
    proxy_supernets = []
    for i, cfg in enumerate(cfgs):
        proxy_supernet = AttentiveNasDynamicModel()
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


if __name__ == "__main__":
    for i in range(100):
        supernet = AttentiveNasDynamicModel()
        subnet, subnet_config = sample_subnet_wo_config(supernet)
        current_subnet = subnet
        payload = current_subnet.cpu().state_dict()
        model = sample_subnet_w_config(AttentiveNasDynamicModel(), subnet_config, False)
        model.load_state_dict(payload)
