"""This submodule controls the different cases of federated learning that could be attacked."""

from .servers import construct_server
from .users import construct_user
from .models import construct_model
from .data import construct_dataloader

__all__ = ["construct_server", "construct_user", "construct_model", "construct_case", "construct_dataloader"]


import torch


def construct_case(cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    """This is a helper function that summarizes the startup, but I find the full protocol to often be clearer."""
    model, loss_fn = construct_model(pretrained=cfg_case.server.pretrained)
    # Server:
    server = construct_server(model, loss_fn, cfg_case, setup)
    model = server.vet_model(model)
    # User:
    user = construct_user(model, loss_fn, cfg_case, setup)
    return user, server, model, loss_fn
