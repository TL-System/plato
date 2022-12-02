"""
Implementation of the model operations.

"""

import torch


def reset_all_weights(model: torch.nn.Module) -> None:
    """Reset trainable andd resettable parameters of the input model
    refs:
    - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
    - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
    - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(module: torch.nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    # Applies fn recursively to every submodule see:
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
