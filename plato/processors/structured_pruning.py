"""
Processor for structured pruning of model weights.
"""

import logging
import copy
import torch
import torch.nn.utils.prune as prune
from plato.processors import model


class Processor(model.Processor):
    """
    A processor for the structured pruning of model weights.
    """

    def __init__(self,
                 parameter_to_prune='weight',
                 conv_dim=0,
                 linear_dim=-1,
                 norm=1,
                 amount=0.2,
                 keep_model=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.parameter_to_prune = parameter_to_prune
        self.conv_dim = conv_dim
        self.linear_dim = linear_dim
        self.norm = norm
        self.amount = amount
        self.keep_model = keep_model

        self.model = self.trainer.model

    def process(self, data):
        """
        Processes structured pruning of model weights layer by layer.
        """
        if self.model is None:
            return data

        if self.keep_model:
            original_state_dict = copy.deepcopy(self.model.cpu().state_dict())

        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, self.parameter_to_prune,
                                    self.amount, self.norm, self.conv_dim)
                prune.remove(module, self.parameter_to_prune)
            elif isinstance(module, torch.nn.Linear):
                prune.ln_structured(module, self.parameter_to_prune,
                                    self.amount, self.norm, self.linear_dim)
                prune.remove(module, self.parameter_to_prune)

        output = self.model.cpu().state_dict()

        if self.keep_model:
            self.model.load_state_dict(original_state_dict)

        if self.client_id is None:
            logging.info("[Server #%d] Structured pruning applied.",
                         self.server_id)
        else:
            logging.info("[Client #%d] Structured pruning applied.",
                         self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """
        Structured pruning applied to a single layer.
        """
        layer_shape = layer.shape()
        if len(layer_shape) <= 1:
            return layer
        else:
            return prune.ln_structured(layer, self.parameter_to_prune,
                                       self.amount, self.norm)
