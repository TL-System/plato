"""
Implements a Processor for global unstructured pruning of model weights.
"""

import logging
from typing import Any

import torch
import torch.nn.utils.prune as prune

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for global unstructured pruning of model weights.
    """

    def __init__(
        self,
        parameters_to_prune=None,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.parameters_to_prune = parameters_to_prune
        self.pruning_method = pruning_method
        self.amount = amount
        self.model = None

    def process(self, data: Any) -> Any:
        """
        Proceesses global unstructured pruning on model weights.
        """

        self.model = self.trainer.model

        if self.parameters_to_prune is None:
            self.parameters_to_prune = []
            for _, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(
                    module, torch.nn.Linear
                ):
                    self.parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=self.pruning_method,
            amount=self.amount,
        )

        for module, name in self.parameters_to_prune:
            prune.remove(module, name)

        output = self.model.cpu().state_dict()

        if self.client_id is None:
            logging.info(
                "[Server #%d] Global unstructured pruning applied.",
                self.server_id,
            )
        else:
            logging.info(
                "[Client #%d] Global unstructured pruning applied.",
                self.client_id,
            )

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        return layer
