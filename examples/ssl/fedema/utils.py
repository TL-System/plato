"""
Tools used by the FedEMA algorithm.
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
    """Get the difference between two sets of parameters."""
    # Compute the divergence between encoders of local and global models
    l2_distance = 0.0
    for paraml, paramg in zip(parameter_a.items(), parameter_b.items()):
        diff = paraml[1] - paramg[1]
        # Calculate L2 norm and add to the total
        l2_distance += torch.sum(diff**2)

    return l2_distance.sqrt()


def update_parameters_moving_average(
    previous_parameters: dict, current_parameters: dict, beta=0.999
):
    """
    Perform the moving average to update the model.

    The weights is directly a OrderDict containing the
    parameters that will be assigned to the model by using moving
    average.

    The hyper-parameters ξ ← βξ + (1 − β)θ

        The most important parameter beta in the moving average
          update method.
        beta:
          - 1: maintain the previous model without being updated with the
              latest weights
          - 0: replace the previous model with the latest weights directly.

    With the increase of the beta, the importance of previous model will increase.
    """

    for parameter_name in previous_parameters:
        if previous_parameters[parameter_name] is not None:
            current_parameters[parameter_name] = (
                previous_parameters[parameter_name] * beta
                + (1 - beta) * current_parameters[parameter_name]
            )

    return current_parameters
