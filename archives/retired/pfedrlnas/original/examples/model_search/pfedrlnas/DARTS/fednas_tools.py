"""
Implement new algorithm: personalized federarted NAS.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from Darts.genotypes import PRIMITIVES

# Exactly the same as search space in DARTS in search phase: 14 edges each with 8 candidates
NUM_EDGES = 14
NUM_OPS = len(PRIMITIVES)


def extract_index(mask):
    """Exract the index with which operation is selected on each edge."""
    result = []
    for i in range(NUM_EDGES):
        edge = mask[i]
        idx = edge.index(1)
        result.append(idx)
    return result


def sample_mask(alphas):
    """Sample a mask for generating the subnet, with a given alpha."""
    mask_pool = []

    for i in range(NUM_OPS):
        mask = np.zeros(NUM_OPS)
        mask[i] = 1
        mask_pool.append(mask)
    result = []

    for alpha in alphas:
        prob = F.softmax(alpha, dim=0)
        prob = prob.detach().numpy()
        prob /= prob.sum()  # make sum = 1
        mask_idx = np.random.choice(range(NUM_OPS), 1, replace=False, p=prob)
        result.append(mask_pool[mask_idx[0]])

    result = np.vstack(result)

    return result.tolist()


def uniform_sample_mask(alphas):
    """Sample a mask, but the probability is uniformed."""
    mask_pool = []

    for i in range(NUM_OPS):
        mask = np.zeros(NUM_OPS)
        mask[i] = 1
        mask_pool.append(mask)

    result = []

    for i in range(len(alphas)):
        mask_idx = np.random.choice(range(NUM_OPS), 1, replace=False)
        result.append(mask_pool[mask_idx[0]])

    result = np.vstack(result)

    return torch.Tensor(result)


def client_weight_param(global_model, client_model):
    """Assign the weights from supernet to subnet."""
    real_ops = []

    for cell_idx, cell in enumerate(client_model.cells):
        for edge_idx, cell_op in enumerate(cell.ops):
            real_ops.append(copy.deepcopy(cell_op.ops))
            cell_op.ops = copy.deepcopy(global_model.cells[cell_idx].ops[edge_idx].ops)

    # copy parameters
    client_model.load_state_dict(global_model.state_dict())

    real_ops_idx = 0

    # only keep selected op
    for cell_idx, cell in enumerate(client_model.cells):
        for edge_idx, cell_op in enumerate(cell.ops):
            if cell.reduction:
                op_mask = client_model.mask_reduce[edge_idx]
            else:
                op_mask = client_model.mask_normal[edge_idx]

            primitive_idx = op_mask.tolist().index(1)
            real_ops[real_ops_idx][0].load_state_dict(
                cell_op.ops[primitive_idx].state_dict()
            )

            cell_op.ops = copy.deepcopy(real_ops[real_ops_idx])
            real_ops_idx += 1


def _average_fuse(global_iter, client_iters, num_samples):
    total_samples = sum(num_samples)

    try:
        while True:
            _, global_param = next(global_iter)
            baseline = global_param.data
            deltas = torch.zeros(baseline.size())
            is_update = torch.zeros(deltas.size())
            for i, client_iter in enumerate(client_iters):
                _, client_param = next(client_iter)
                delta = client_param.data - baseline
                is_update = torch.ones(is_update.size()) - (
                    torch.abs(delta) < 1e-8
                ).type(dtype=is_update.dtype)
                deltas += delta * num_samples[i] / total_samples * is_update
            global_param.data += deltas
    except StopIteration:
        pass


def fuse_weight_gradient(global_model, client_models, num_samples):
    """Fuse weights of subnets with different structure into supernet"""
    proxy_client_models = []

    for client_model in client_models:
        proxy_client = copy.deepcopy(client_model)
        for cell_idx, cell in enumerate(client_model.cells):
            for edge_idx, cell_op in enumerate(cell.ops):
                if cell.reduction:
                    op_mask = client_model.mask_reduce[edge_idx]
                else:
                    op_mask = client_model.mask_normal[edge_idx]
                primitive_idx = op_mask.tolist().index(1)
                proxy_client.cells[cell_idx].ops[edge_idx] = copy.deepcopy(
                    global_model.cells[cell_idx].ops[edge_idx]
                )
                proxy_client.cells[cell_idx].ops[edge_idx].ops[
                    primitive_idx
                ].load_state_dict(cell_op.ops[0].state_dict())
        proxy_client_models.append(proxy_client)

    proxy_iters = []

    for proxy_supernet in proxy_client_models:
        proxy_iters.append(proxy_supernet.named_parameters())

    global_iter = global_model.named_parameters()
    _average_fuse(global_iter, proxy_iters, num_samples)
