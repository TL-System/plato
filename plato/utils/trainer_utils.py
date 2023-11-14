"""
The necessary tools used by trainers.
"""


def freeze_model(model, layer_names=None):
    """Freeze a part of the model."""
    if layer_names is not None:
        frozen_params = []
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = False
                frozen_params.append(name)


def activate_model(model, layer_names=None):
    """Activate a part of the model."""
    if layer_names is not None:
        for name, param in model.named_parameters():
            if any(param_name in name for param_name in layer_names):
                param.requires_grad = True
