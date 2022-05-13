"""
Implements a Processor for global pruning of model weights.
"""
import logging
from typing import Any
import copy

import torch
import torch.nn.utils.prune as prune

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for global pruning of model weights.
    """
    def __init__(self,
                 parameters_to_prune=[],
                 pruning_method=prune.L1Unstructured,
                 amount=0.2,
                 keep_model=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.parameters_to_prune = parameters_to_prune
        self.pruning_method = pruning_method
        self.amount = amount
        self.keep_model = keep_model

        self.model = self.trainer.model

        if len(self.parameters_to_prune) == 0:
            for _, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(
                        module, torch.nn.Linear):
                    self.parameters_to_prune.append((module, 'weight'))

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for global pruning of model weights.
        """

        if self.model is None:
            return data

        if self.keep_model:
            original_state_dict = copy.deepcopy(self.model.cpu().state_dict())

        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=self.pruning_method,
            amount=self.amount,
        )

        for module, name in self.parameters_to_prune:
            prune.remove(module, name)

        output = self.model.cpu().state_dict()

        if self.keep_model:
            self.model.load_state_dict(original_state_dict)

        logging.info("[Client #%d] Global pruning applied.", self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        return layer
