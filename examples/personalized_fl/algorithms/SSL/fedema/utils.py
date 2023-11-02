"""
Tools used in algorithm FedEMA
"""
from collections import OrderedDict
import torch


def extract_encoder(model_layers, encoder_layer_names):
    """Extract the encoder layers from the model layers."""
    return OrderedDict(
        [
            (name, param)
            for name, param in model_layers.items()
            if any(
                param_name in name.strip().split(".")
                for param_name in encoder_layer_names
            )
        ]
    )


def get_parameters_diff(parameter_a: OrderedDict, parameter_b: OrderedDict):
    """Get the difference between two sets of parameters"""
    # Compute the divergence between encoders of local and global models
    l2_distance = 0.0
    for paraml, paramg in zip(parameter_a.items(), parameter_b.items()):
        diff = paraml[1] - paramg[1]
        # Calculate L2 norm and add to the total
        l2_distance += torch.sum(diff**2)

    return l2_distance.sqrt()
