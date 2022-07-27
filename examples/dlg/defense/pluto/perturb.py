import math

import numpy as np
import torch
import torch.nn as nn
from plato.config import Config
from plato.utils import csv_processor
from utils.pseudorandom import getGause

csv_file = f"var.csv"

def compute_risk(model: nn.Module):
    var = []
    for name, param in model.named_parameters():
        var.append(torch.var(param).detach().numpy())

    csv_processor.write_csv(csv_file, var)

    # Normalize values in [0,1]
    norm_var = [(float(i)-min(var))/(max(var)-min(var)) for i in var]
    return norm_var


def noise(dy_dx: list, risk: list):
    for i in range(len(dy_dx)):
        grad_tensor = dy_dx[i].cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(
            flattened_weights, Config().algorithm.prune_pct)
        grad_tensor = np.where(
            abs(grad_tensor) < thresh, 0, grad_tensor)
        grad_tensor += getGause(scale=risk[i]*Config().algorithm.perturb_base)
        dy_dx[i] = torch.Tensor(
            grad_tensor).to(Config().device())
    
    return dy_dx