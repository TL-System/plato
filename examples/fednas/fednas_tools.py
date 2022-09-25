import torch
import numpy as np
import torch.nn.functional as F
import os
import sys
import time
import glob
import torch
import Darts.Dartsutils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
import random

from torch.autograd import Variable
from Darts.model_search import Network
from Darts.model_search_local import MaskedNetwork
from Darts.architect import Architect

num_edges = 14
num_ops = 8
grad_clip = 5

def extract_index(mask):
    result = []
    for i in range(num_edges):
        edge = mask[i].tolist()
        idx = edge.index(1)
        result.append(idx)
    return result

def _extract_pos(name):
    if 'stem' in name:
        return (-1,0,0)
    if 'classifier' in name:
        return (-2,0,0)
    name_list = name.split('.')
    # cells.0._ops.0._ops.4.op.1.weight
    cell_idx = int(name_list[1])
    if 'preprocess0' in name:
        edge_idx = -1
        op_idx = 0
    elif 'preprocess1' in name:
        edge_idx = -2
        op_idx = 0
    else:
        edge_idx = int(name_list[3])
        op_idx = int(name_list[5])
    pos = (cell_idx, edge_idx, op_idx)
    return pos


# todo: ? None operation
def sample_mask(alphas):
    mask_pool = []
    for i in range(num_ops):
        mask = np.zeros(num_ops)
        mask[i] = 1
        mask_pool.append(mask)
    result = []
    for i in range(len(alphas)):
        prob = F.softmax(alphas[i],dim=0)
        prob = prob.tolist()
        prob[0] += (1-sum(prob)) # add precision loss, make sum = 1
        mask_idx = np.random.choice([i for i in range(num_ops)],1,replace=False,p=prob)
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

def _assign_client_param(global_param,client_params):
    for i in range(len(client_params)):
        client_params[i].data = global_param.data

def _assign_client(global_iter,client_iters):
    global_name, global_param = next(global_iter)
    for i in range(len(client_iters)):
        client_name,client_param = next(client_iters[i])
        client_param.data = global_param.data

def client_weight_param(global_model, client_models):
    mask_normal = torch.zeros(num_edges, num_ops)
    mask_reduce = torch.zeros(num_edges, num_ops)
    index_normal = []
    index_reduce = []
    client_iters = []
    model=client_models
    mask_normal = mask_normal + model.mask_normal
    mask_reduce = mask_reduce + model.mask_reduce
    index_normal.append(extract_index(model.mask_normal))
    index_reduce.append(extract_index(model.mask_reduce))
    client_iters.append(model.named_parameters())
    global_iter = global_model.named_parameters()
    # num of paramsa
    # fuse gradient of stem
    for k in range(3):
        _assign_client(global_iter, client_iters)

    # fuse gradient of preprocess
    for k in range(2):
        _assign_client(global_iter, client_iters)

    # fuse gradient of cells
    client_positions = [None]
    client_params = [None]

    client_name, client_param = next(client_iters[0])
    client_pos = _extract_pos(client_name)
    client_positions[0] = client_pos
    client_params[0] = client_param

    last_cell_idx = 0
    while True:
        global_name, global_param = next(global_iter)
        # print("global name", global_name, 'param shape', global_param.size())
        if 'classifier' in global_name:
            break
        global_pos = _extract_pos(global_name)
        cell_idx, edge_idx, op_idx = global_pos
        # fuse gradient of preprocess for each cell
        if cell_idx != last_cell_idx:
            last_cell_idx = cell_idx
            _assign_client_param(global_param, client_params)
            _assign_client(global_iter, client_iters)
            if cell_idx in [3, 6]:  # extra preprocess after reduction
                _assign_client(global_iter, client_iters)
            for i in range(len([client_models])):
                client_name, client_param = next(client_iters[i])
                client_pos = _extract_pos(client_name)
                client_positions[i] = client_pos
                client_params[i] = client_param
            global_name, global_param = next(global_iter)
            # print("global name", global_name, 'param shape', global_param.size())
            global_pos = _extract_pos(global_name)
            cell_idx, edge_idx, op_idx = global_pos
        if cell_idx in [0, 1, 3, 4, 6, 7]:  # normal cell
            index = index_normal
            mask = mask_normal
        else:  # reduce cell
            index = index_reduce
            mask = mask_reduce
        num_client = int(mask[edge_idx][op_idx])
        if num_client != 0:
            # sum gradients
            for client_idx in range(len([client_models])):
                client_pos = client_positions[client_idx]
                if index[client_idx][edge_idx] == global_pos[2] and client_pos[0] == global_pos[0] and client_pos[1] == \
                        client_pos[1]:
                    # print("client pos", client_pos,'gate idx', index[client_idx][edge_idx], 'param shape', client_params[client_idx].size())
                    client_params[client_idx].data = global_param.data
                    # next iter of client model
                    client_name, client_param = next(client_iters[client_idx])
                    client_pos = _extract_pos(client_name)
                    client_positions[client_idx] = client_pos
                    client_params[client_idx] = client_param

    # fuse gradient of classifier
    _assign_client_param(global_param, client_params)
    _assign_client(global_iter, client_iters)
    return

def _average_fuse(global_iter,client_iters):
    global_name, global_param = next(global_iter)
    baseline=global_param.data
    deltas=baseline-global_param.data
    for i in range(len(client_iters)):
        client_name, client_param = next(client_iters[i])
        deltas+=client_param.data-baseline
    global_param.data = baseline+deltas/len(client_iters)

def _average_fuse_param(global_param,client_params):
    baseline = global_param.data
    deltas = baseline - global_param.data
    for i in range(len(client_params)):
        client_param = client_params[i]
        deltas+=client_param.data-baseline
    global_param.data = baseline+deltas/len(client_params)


def fuse_weight_gradient(global_model, client_models):
    mask_normal = torch.zeros(num_edges,num_ops)
    mask_reduce = torch.zeros(num_edges, num_ops)
    index_normal = []
    index_reduce = []
    client_iters = []
    for model in client_models:
        mask_normal = mask_normal + model.mask_normal
        mask_reduce = mask_reduce + model.mask_reduce
        index_normal.append(extract_index(model.mask_normal))
        index_reduce.append(extract_index(model.mask_reduce))
        client_iters.append(model.named_parameters())
    global_iter = global_model.named_parameters()
    # num of params
    # stem: 3, normal cell: 170, reduce cell: 186, classifier: 2

    # fuse gradient of stem
    for k in range(3):
        _average_fuse(global_iter,client_iters)

    # fuse gradient of preprocess
    for k in range(2):
        _average_fuse(global_iter, client_iters)

    # fuse gradient of cells
    client_positions = [None for _ in range(len(client_models))]
    client_params = [None for _ in range(len(client_models))]
    for i in range(len(client_models)):
        client_name, client_param = next(client_iters[i])
        client_pos = _extract_pos(client_name)
        client_positions[i] = client_pos
        client_params[i] = client_param

    last_cell_idx = 0
    while True:
        global_name, global_param = next(global_iter)
        # print("global name", global_name, 'param shape', global_param.size())
        if 'classifier' in global_name:
            break
        global_pos = _extract_pos(global_name)
        cell_idx, edge_idx, op_idx = global_pos
        # fuse gradient of preprocess for each cell
        if cell_idx != last_cell_idx:
            last_cell_idx = cell_idx
            _average_fuse_param(global_param,client_params)
            _average_fuse(global_iter, client_iters)
            if cell_idx in [3,6]: # extra preprocess after reduction
                _average_fuse(global_iter, client_iters)
            for i in range(len(client_models)):
                client_name, client_param = next(client_iters[i])
                client_pos = _extract_pos(client_name)
                client_positions[i] = client_pos
                client_params[i] = client_param
            global_name, global_param = next(global_iter)
            # print("global name", global_name, 'param shape', global_param.size())
            global_pos = _extract_pos(global_name)
            cell_idx, edge_idx, op_idx = global_pos
        if cell_idx in [0,1,3,4,6,7]: # normal cell
            index = index_normal
            mask = mask_normal
        else: # reduce cell
            index = index_reduce
            mask = mask_reduce
        num_client = int(mask[edge_idx][op_idx])
        if num_client != 0:
            # sum gradients
            baseline=global_param.data
            deltas=baseline-global_param.data
            for client_idx in range(len(client_models)):
                client_pos = client_positions[client_idx]
                if index[client_idx][edge_idx] == global_pos[2] and client_pos[0] == global_pos[0] and client_pos[1] == client_pos[1]:
                    # print("client pos", client_pos,'gate idx', index[client_idx][edge_idx], 'param shape', client_params[client_idx].size())
                    deltas+=client_params[client_idx].data-baseline
                    # next iter of client model
                    client_name, client_param = next(client_iters[client_idx])
                    client_pos = _extract_pos(client_name)
                    client_positions[client_idx] = client_pos
                    client_params[client_idx] = client_param
            # average gradients
            global_param.data = baseline+deltas / num_client

    # fuse gradient of classifier
    _average_fuse_param(global_param, client_params)
    _average_fuse(global_iter, client_iters)
    return

def stale_generate(num,stale):
    array=[]
    prob_mark=[0.7,0.4,0.1]
    #[0.4,0.05,0.01]
    #[0.1,0.01,0.001]
    for i in range(num):
        prob=random.random()
        if prob>=prob_mark[0]:#0.1:
            array.append(0)
        elif prob>=prob_mark[1]:#0.01:
            array.append(1)
        elif prob>=prob_mark[2]:#0.001:
            array.append(2)
        else:
            array.append(stale+1)
    return np.array(array)

def init_gradient(model):
    print("initializing gradient")
    data = Variable(torch.Tensor(1,3,32,32),requires_grad=False)
    logits = model(data)
    loss = torch.mean(logits)
    loss.backward()
    model.zero_grad()
    model.alphas_normal.grad = torch.zeros(model.alphas_normal.size())
    model.alphas_reduce.grad = torch.zeros(model.alphas_reduce.size())
    return




if __name__ == '__main__':
    alphas = torch.Tensor(14,8)
    result = uniform_sample_mask(alphas)
    print(result)

