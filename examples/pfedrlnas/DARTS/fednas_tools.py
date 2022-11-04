import torch
import numpy as np
import torch.nn.functional as F
import os
import sys
import time
import glob
import torch
import Darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
import random
from collections import OrderedDict
import logging

from torch.autograd import Variable
from Darts.model_search import Network
from Darts.model_search_local import MaskedNetwork
from Darts.architect import Architect
from Darts.genotypes import PRIMITIVES

num_edges = 14
num_ops = len(PRIMITIVES)


def extract_index(mask):
    result = []
    for i in range(num_edges):
        edge = mask[i].tolist()
        idx = edge.index(1)
        result.append(idx)
    return result


def _extract_pos(name):
    if "stem" in name:
        return (-1, 0, 0)
    if "classifier" in name:
        return (-2, 0, 0)
    name_list = name.split(".")
    # cells.0._ops.0._ops.4.op.1.weight
    cell_idx = int(name_list[1])
    if "preprocess0" in name:
        edge_idx = -1
        op_idx = 0
    elif "preprocess1" in name:
        edge_idx = -2
        op_idx = 0
    else:
        edge_idx = int(name_list[3])
        op_idx = int(name_list[5])
    pos = (cell_idx, edge_idx, op_idx)
    return pos


# todo: ? None operation
def sample_mask(
    alphas,
):
    mask_pool = []
    for i in range(num_ops):
        mask = np.zeros(num_ops)
        mask[i] = 1
        mask_pool.append(mask)
    result = []
    for i in range(len(alphas)):
        prob = F.softmax(alphas[i], dim=0)
        prob = prob.detach().numpy()
        prob /= prob.sum()
        mask_idx = np.random.choice(
            [i for i in range(num_ops)], 1, replace=False, p=prob
        )
        result.append(mask_pool[mask_idx[0]])
    result = np.vstack(result)
    return torch.Tensor(result)


def uniform_sample_mask(alphas):
    mask_pool = []
    for i in range(num_ops):
        mask = np.zeros(num_ops)
        mask[i] = 1
        mask_pool.append(mask)
    result = []
    for i in range(len(alphas)):
        mask_idx = np.random.choice([i for i in range(num_ops)], 1, replace=False)
        result.append(mask_pool[mask_idx[0]])
    result = np.vstack(result)
    return torch.Tensor(result)


def client_weight_param(global_model, client_model):
    # expand client model as the same as global model
    real_ops = []
    for cell_idx, cell in enumerate(client_model.cells):
        for edge_idx in range(len(cell._ops)):
            real_ops.append(
                copy.deepcopy(client_model.cells[cell_idx]._ops[edge_idx]._ops)
            )
            client_model.cells[cell_idx]._ops[edge_idx]._ops = copy.deepcopy(
                global_model.cells[cell_idx]._ops[edge_idx]._ops
            )

    # copy parameters
    client_model.load_state_dict(global_model.state_dict())

    real_ops_idx = 0
    # only keep selected op
    for cell_idx, cell in enumerate(client_model.cells):
        for edge_idx in range(len(cell._ops)):
            if cell.reduction:
                op_mask = client_model.mask_reduce[edge_idx]
            else:
                op_mask = client_model.mask_normal[edge_idx]
            primitive_idx = op_mask.tolist().index(1)
            real_ops[real_ops_idx][0].load_state_dict(
                client_model.cells[cell_idx]
                ._ops[edge_idx]
                ._ops[primitive_idx]
                .state_dict()
            )
            client_model.cells[cell_idx]._ops[edge_idx]._ops = copy.deepcopy(
                real_ops[real_ops_idx]
            )
            real_ops_idx += 1


def _average_fuse(global_iter, client_iters, num_samples, avg_last=True):
    # total_samples=sum(num_samples)
    try:
        while True:
            global_name, global_param = next(global_iter)
            if (not avg_last) and ("classifier" in global_name):
                for i in range(len(client_iters)):
                    client_name, client_param = next(client_iters[i])
                continue
            baseline = copy.deepcopy(global_param.data)
            deltas = torch.zeros(baseline.size())
            is_updates = torch.zeros(deltas.size())
            for i in range(len(client_iters)):
                client_name, client_param = next(client_iters[i])
                delta = client_param.data - baseline
                is_update = torch.where(
                    client_param.data == baseline,
                    torch.zeros(delta.size()),
                    torch.ones(delta.size()),
                )
                deltas += delta * num_samples[i] * is_update
                is_updates += is_update * num_samples[i]
            is_updates = torch.where(
                is_updates == 0, torch.ones(is_updates.size()), is_updates
            )
            # logging.info(is_updates)
            deltas = torch.div(deltas, is_updates)
            global_param.data += deltas
    except StopIteration:
        pass


def fuse_weight_gradient(global_model, client_models, num_samples, avg_last=True):
    proxy_client_models = []
    for client_idx in range(len(client_models)):
        proxy_client = copy.deepcopy(client_models[client_idx])
        for cell_idx, cell in enumerate(client_models[client_idx].cells):
            for edge_idx in range(len(cell._ops)):
                if cell.reduction:
                    op_mask = client_models[client_idx].mask_reduce[edge_idx]
                else:
                    op_mask = client_models[client_idx].mask_normal[edge_idx]
                primitive_idx = op_mask.tolist().index(1)
                proxy_client.cells[cell_idx]._ops[edge_idx] = copy.deepcopy(
                    global_model.cells[cell_idx]._ops[edge_idx]
                )
                proxy_client.cells[cell_idx]._ops[edge_idx]._ops[
                    primitive_idx
                ].load_state_dict(
                    client_models[client_idx]
                    .cells[cell_idx]
                    ._ops[edge_idx]
                    ._ops[0]
                    .state_dict()
                )
        proxy_client_models.append(proxy_client)

    proxy_iters = []
    for proxy_supernet in proxy_client_models:
        proxy_iters.append(proxy_supernet.named_parameters())
    global_iter = global_model.named_parameters()
    _average_fuse(global_iter, proxy_iters, num_samples, avg_last)


def stale_generate(num, stale):
    array = []
    prob_mark = [0.7, 0.4, 0.1]
    # [0.4,0.05,0.01]
    # [0.1,0.01,0.001]
    for i in range(num):
        prob = random.random()
        if prob >= prob_mark[0]:  # 0.1:
            array.append(0)
        elif prob >= prob_mark[1]:  # 0.01:
            array.append(1)
        elif prob >= prob_mark[2]:  # 0.001:
            array.append(2)
        else:
            array.append(stale + 1)
    return np.array(array)


def init_gradient(model, C):
    # print("initializing gradient")
    data = Variable(torch.Tensor(C, 3, 32, 32), requires_grad=False)
    logits = model(data)
    loss = torch.mean(logits)
    loss.backward()
    model.zero_grad()
    model.alphas_normal.grad = torch.zeros(model.alphas_normal.size())
    model.alphas_reduce.grad = torch.zeros(model.alphas_reduce.size())
    return


if __name__ == "__main__":
    alphas = torch.Tensor(14, 8)
    result = uniform_sample_mask(alphas)
    print(result)
