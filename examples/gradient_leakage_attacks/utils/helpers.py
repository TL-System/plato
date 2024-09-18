"""Helper methods"""

import torch
import torch.nn.functional as F


def label_to_onehot(target, num_classes=10):
    """Convert labels to one-hot."""
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    """Calculate cross entropy loss for one-hot labels."""
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
