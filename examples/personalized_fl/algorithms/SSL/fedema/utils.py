"""
Tools to compute the model statistic.
"""
from collections import OrderedDict


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


def get_model_statistic(model_parameters):
    """Getting the model statistic."""

    parameters_statistic = {}
    for name, parameters in model_parameters.items():
        parameters_statistic[name] = {
            "mean": parameters.mean(),
            "std": parameters.std(),
            "max": parameters.max(),
            "min": parameters.min(),
        }
    model_statistic = {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    for name, value in parameters_statistic.items():
        model_statistic["mean"] += value["mean"]
        model_statistic["std"] += value["std"]
        model_statistic["max"] += value["max"]
        model_statistic["min"] += value["min"]

    return model_statistic
