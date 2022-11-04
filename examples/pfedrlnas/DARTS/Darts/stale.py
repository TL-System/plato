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

from torch.autograd import Variable
from .model_search import Network
from .model_search_local import MaskedNetwork


def compute_stale_grad_weight(old_model, new_model):
    # old_model=old_model.cuda()
    # new_model=new_model.cuda()
    old_weight_iter = old_model.parameters()
    new_weight_iter = new_model.parameters()
    try:
        while True:
            old_weight = next(old_weight_iter)
            old_weight.cuda()
            old_grad = old_weight.grad
            new_weight = next(new_weight_iter)
            new_weight.cuda()
            # multiplier lambda = 1
            approx_2nd_grad = old_grad * old_grad * (new_weight - old_weight)
            old_weight.grad += approx_2nd_grad
            old_weight.cpu()
            new_weight.cpu()
    except StopIteration:
        pass
    # old_model=old_model.cpu()
    # new_model=new_model.cpu()
    return


def compute_stale_grad_alpha(index_list, old_alphas, new_alphas):
    old_prob = F.softmax(old_alphas, -1)
    # multiplier lambda = 1
    result_grad = old_prob + old_prob * old_prob * (new_alphas - old_alphas)
    for edge_idx in range(old_alphas.shape[0]):
        op_idx = index_list[edge_idx]
        i_prob = old_prob[edge_idx][op_idx] - 1
        i_new_alpha = new_alphas[edge_idx][op_idx]
        i_old_alpha = old_alphas[edge_idx][op_idx]
        result_grad[edge_idx][op_idx] = i_prob + i_prob * i_prob * (
            i_new_alpha - i_old_alpha
        )
    return result_grad


if __name__ == "__main__":
    from torchvision import models

    # old_model = models.vgg11()
    # data = torch.ones(1,3,32,32)
    # output = old_model(data)
    # loss = torch.mean(output)
    # loss.backward()
    # new_model = models.vgg11(pretrained=True)
    #
    # old_param = copy.deepcopy(next(old_model.parameters()))
    #
    # compute_stale_grad_weight(old_model,new_model)
    # new_param = next(old_model.parameters())
    # if (old_param==new_param).all():
    #     print("fail")
    # else:
    #     print("pass")

    old_alphas = torch.rand(14, 8)
    index_list = []
    for i in range(14):
        index_list.append(i // 2)
    new_alphas = torch.rand(14, 8)
    old_alphas.grad = torch.zeros(14, 8)
    result = compute_stale_grad_alpha(index_list, old_alphas, new_alphas)
    print(result)
