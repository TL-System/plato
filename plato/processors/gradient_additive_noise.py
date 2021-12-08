"""
A Processor of differential privacy to clip and add noise on gradients of model weights.
"""

from collections import OrderedDict
import logging
import os
from typing import Any
import torch
import numpy as np

from plato.processors import base


class Processor(base.Processor):
    """
    Implement a Processor to clip and add noise on gradients.
    """
    def __init__(self, client_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

        # Save the model weights before training to get gradients
        # for adding noise to ensure differential privacy
        self.previous_model_weights = None

    def process(self, data: Any) -> Any:
        """
        Clip and add noise on gradients to guarantee differential privacy.
        """
        # Server does not add noise in the first training iteration
        if self.previous_model_weights is None:
            return data

        gradients = OrderedDict()

        print("in process()")
        print("Previous model weights:")
        print(self.previous_model_weights)
        print("New model weights:")
        print(data)
        for (name, new_weight), (__, previous_weight) in zip(
                data.items(), self.previous_model_weights.items()):
            # Compute gradients
            gradients[name] = new_weight - previous_weight

        clipped_gradients, clipping_bound = Processor.clip_gradients(gradients)

        processed_weights = OrderedDict()

        for (name, old_weight), (_, clipped_gradient) in zip(
                self.previous_model_weights.items(),
                clipped_gradients.items()):
            # Add noise to updated weights
            processed_weights[
                name] = old_weight + clipped_gradient + self.compute_additive_noise(
                    clipped_gradient, clipping_bound)

        if self.client_id is None:
            logging.info("[Server #%d] Applied local differential privacy.",
                         os.getpid())
        else:
            logging.info("[Client #%d] Applied local differential privacy.",
                         self.client_id)

        return processed_weights

    @staticmethod
    def clip_gradients(gradients):
        """ Clips the gradients in preparation for adding noise. """
        clipped_gradients = OrderedDict()

        norm_list = [
            torch.linalg.norm(gradient.float()).item()
            for _, gradient in gradients.items()
        ]
        # Set the clipping bound as the median of the L2 norm of gradients
        clipping_bound = np.median(norm_list)

        # Compute clipped gradients
        for name, gradient in gradients.items():
            gradient_norm = torch.linalg.norm(gradient.float()).item()

            if clipping_bound == 0:
                clipped_gradients[name] = gradient
            else:
                clipped_gradients[name] = gradient / max(
                    1, gradient_norm / clipping_bound)

        return clipped_gradients, clipping_bound
