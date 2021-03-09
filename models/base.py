"""Class definition for Plato models."""

import torch
from config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    import mindspore
    Model = mindspore.nn.Cell
else:
    Model = torch.nn.Module
