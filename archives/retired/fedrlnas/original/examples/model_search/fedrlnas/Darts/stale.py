"""
A tool for Soft Synchronization in FedRLNAS.

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522
"""

import torch.nn.functional as F


def compute_stale_grad_weight(old_model, new_model):
    """Use Async SGD + Talor Compensation to aggregate stale weights"""
    old_weight_iter = old_model.parameters()
    new_weight_iter = new_model.parameters()

    try:
        while True:
            old_weight = next(old_weight_iter)
            old_weight.cuda()
            old_grad = old_weight.grad
            new_weight = next(new_weight_iter)
            new_weight.cuda()
            approx_2nd_grad = old_grad * old_grad * (new_weight - old_weight)
            old_weight.grad += approx_2nd_grad
            old_weight.cpu()
            new_weight.cpu()
    except StopIteration:
        pass


def compute_stale_grad_alpha(index_list, old_alphas, new_alphas):
    """Use Async SGD + Talor Compensation to update stale alphas (structure parameter)."""
    old_prob = F.softmax(old_alphas, -1)
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
