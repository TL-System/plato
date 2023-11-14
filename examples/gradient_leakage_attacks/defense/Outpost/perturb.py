import numpy as np
import torch
import torch.nn as nn
from plato.config import Config


def compute_risk(model: nn.Module):
    var = []
    for param in model.parameters():
        var.append(torch.var(param).cpu().detach().numpy())
    var = [min(v, 1) for v in var]
    return var


def noise(dy_dx: list, risk: list):
    # Calculate empirical FIM
    fim = []
    flattened_fim = None
    for i in range(len(dy_dx)):
        squared_grad = dy_dx[i].clone().pow(2).mean(0).cpu().numpy()
        fim.append(squared_grad)
        if flattened_fim is None:
            flattened_fim = squared_grad.flatten()
        else:
            flattened_fim = np.append(flattened_fim, squared_grad.flatten())

    fim_thresh = np.percentile(flattened_fim, 100 - Config().algorithm.phi)

    for i in range(len(dy_dx)):
        # pruning
        grad_tensor = dy_dx[i].cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, Config().algorithm.prune_base)
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        # noise
        noise_base = torch.normal(
            0, risk[i] * Config().algorithm.noise_base, dy_dx[i].shape
        )
        noise_mask = np.where(fim[i] < fim_thresh, 0, 1)
        gauss_noise = noise_base * noise_mask
        dy_dx[i] = (
            (torch.Tensor(grad_tensor) + gauss_noise)
            .to(dtype=torch.float32)
            .to(Config().device())
        )

    return dy_dx
