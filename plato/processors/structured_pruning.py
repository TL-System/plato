"""
Processor for structured pruning of model weights.
"""

import logging
import torch
import torch.nn.utils.prune as prune
from plato.processors import model


class Processor(model.Processor):
    """
    A processor for the structured pruning of model weights.
    """

    def __init__(self,
                 pruning_method='ln',
                 amount=0.2,
                 norm=1,
                 dim=-1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.pruning_method = pruning_method
        self.amount = amount
        self.norm = norm
        self.dim = dim
        self.model = None

    def process(self, data):
        """
        Processes structured pruning of model weights layer by layer.
        """
        self.model = self.trainer.model

        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                    module, torch.nn.Linear):
                if self.pruning_method == 'ln':
                    prune.ln_structured(module,
                                        'weight',
                                        self.amount,
                                        n=self.norm,
                                        dim=self.dim)
                elif self.pruning_method == 'random':
                    prune.random_structured(module,
                                            'weight',
                                            self.amount,
                                            dim=self.dim)
                prune.remove(module, 'weight')

        output = self.model.cpu().state_dict()

        if self.client_id is None:
            logging.info("[Server #%d] Structured pruning applied.",
                         self.server_id)
        else:
            logging.info("[Client #%d] Structured pruning applied.",
                         self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """ No need to process individual layer of the model """
